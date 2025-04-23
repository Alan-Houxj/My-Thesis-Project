import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse # 用于命令行参数

from environment import PricingEnvironment
from agent import MAPPOAgent, Critic # 导入 MAPPOAgent 和 Critic

def main(args):
    """主训练函数 (RNN 版本)"""

    # --- 设置 ---
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 创建结果保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --- 初始化环境 (传入所有参数) ---
    # 检查 omega 范围
    if not (0.0 <= args.omega <= 1.0):
         print(f"警告: Omega 值 ({args.omega}) 不在 [0, 1] 范围内，将强制设为 1.0 (纯个体奖励)")
         args.omega = 1.0

    env = PricingEnvironment(
        n_products=args.n_agents,
        n_price_levels=args.n_price_levels,
        episode_length=args.episode_len,
        homogeneous=args.homogeneous,
        shock_prob=args.shock_prob,
        shock_duration=args.shock_duration,
        shock_magnitude=args.shock_magnitude,
        initial_inventory=args.initial_inventory,
        holding_cost=args.holding_cost,
        replenishment_threshold=args.replenishment_threshold,
        replenishment_amount=args.replenishment_amount,
        max_inventory_capacity=args.max_inventory_capacity,
        replenishment_lead_time=args.replenishment_lead_time,
        stockout_penalty_per_unit=args.stockout_penalty,
        omega=args.omega
    )
    env_info = env.get_env_info()
    print("环境信息 (RNN Actor):", env_info)

    # --- 初始化智能体和共享 Critic ---
    shared_critic = Critic(env_info['state_shape'], hidden_dim=args.critic_hidden_dim).to(device)
    shared_critic_optimizer = torch.optim.Adam(shared_critic.parameters(), lr=args.critic_lr)

    agents = []
    for agent_id in range(args.n_agents):
        agent = MAPPOAgent(
            obs_dim=env_info['obs_shape'],
            state_dim=env_info['state_shape'],
            action_dim=env_info['n_actions'],
            n_agents=args.n_agents,
            agent_id=agent_id,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_param=args.clip_param,
            entropy_coef=args.entropy_coef,
            update_epochs=args.update_epochs,
            rnn_hidden_dim=args.rnn_hidden_dim,
            device=device
        )
        agents.append(agent)

    # --- 训练循环 ---
    all_episode_rewards = [] # 记录每个 episode 的总奖励（所有智能体）
    avg_rewards_history = [] # 记录平均总奖励变化
    actor_loss_history = [[] for _ in range(args.n_agents)]
    critic_loss_history = []
    # 新增: 记录每个智能体每个周期的原始奖励总和
    raw_agent_rewards_per_episode = [[] for _ in range(args.n_agents)]
    # 新增: 记录每个智能体的平滑平均奖励历史
    agent_avg_rewards_history = [[] for _ in range(args.n_agents)]

    print("\n--- 开始训练 (RNN Actor) ---")
    for episode in range(args.n_episodes):
        # -- 数据收集 (需要存储隐藏状态) --
        episode_data = {
            'obs': [], 'state': [], 'actions': [], 'rewards': [],
            'dones': [], 'old_log_probs': [], 'next_state': [],
            'actor_h_in': [], 'actor_c_in': [], # 存储 Actor 的 LSTM 输入隐藏状态
            'masks': [] # 用于标记有效步骤
        }
        episode_total_reward = 0
        # 修改: step_rewards 现在是 numpy array，便于累加
        episode_agent_rewards = np.zeros(args.n_agents) # 记录本周期每个 agent 的总奖励

        obs_list, state = env.reset()
        # 初始化所有智能体的隐藏状态 (h, c)
        actor_hidden_states = [agent.init_hidden() for agent in agents]

        for t in range(args.episode_len):
            current_actions = []
            current_log_probs = []
            # 修改: 使用 squeeze() 完全去除多余维度
            current_h_in = [h.squeeze().cpu().numpy() for h, c in actor_hidden_states] # 应该得到 (hidden_dim,)
            current_c_in = [c.squeeze().cpu().numpy() for h, c in actor_hidden_states] # 应该得到 (hidden_dim,)

            # 1. 所有智能体根据观察和隐藏状态选择动作
            temp_next_hidden_states = [] # 临时存储下一步的 hidden state
            with torch.no_grad():
                for agent_id in range(args.n_agents):
                    action, log_prob, next_hidden_state = agents[agent_id].choose_action(
                        obs_list[agent_id], actor_hidden_states[agent_id]
                    )
                    current_actions.append(action)
                    current_log_probs.append(log_prob.item())
                    temp_next_hidden_states.append(next_hidden_state) # 存储更新后的隐藏状态

            # 更新隐藏状态供下一步使用
            actor_hidden_states = temp_next_hidden_states

            # 2. 环境执行动作
            next_obs_list, next_state, rewards, dones, info = env.step(current_actions)

            # 3. 存储经验 (包括隐藏状态)
            episode_data['obs'].append(np.array(obs_list))
            episode_data['state'].append(np.array(state))
            episode_data['actions'].append(np.array(current_actions))
            episode_data['rewards'].append(np.array(rewards))
            episode_data['dones'].append(np.array(dones))
            episode_data['old_log_probs'].append(np.array(current_log_probs))
            episode_data['next_state'].append(np.array(next_state))
            # 存储的是动作选择前的隐藏状态
            episode_data['actor_h_in'].append(np.array(current_h_in)) # (n_agents, rnn_hidden_dim)
            episode_data['actor_c_in'].append(np.array(current_c_in)) # (n_agents, rnn_hidden_dim)
            episode_data['masks'].append(np.array([1.0])) # 标记此步有效 (shape (1,))

            # 更新状态
            obs_list = next_obs_list
            state = next_state
            # 修改: 累加每个 agent 的奖励
            episode_agent_rewards += np.array(rewards)
            # episode_total_reward += sum(rewards) # 这个会在后面从 episode_agent_rewards 计算

            if all(dones):
                break

        # 周期结束，计算总奖励
        episode_total_reward = episode_agent_rewards.sum()

        # -- 数据格式化和学习 --
        actual_len = len(episode_data['state'])
        batch = {}
        if actual_len > 0:
            for key, data_list in episode_data.items():
                arr = np.array(data_list) # 先转换为 NumPy 数组

                if key == 'masks':
                    # 特别处理 masks: 期望 (len, 1) -> (1, len, 1)
                    if arr.ndim == 1: arr = arr[:, np.newaxis] # 确保是 (len, 1)
                    if arr.shape != (actual_len, 1):
                         raise ValueError(f"Mask array has unexpected shape: {arr.shape}, expected ({actual_len}, 1)")
                    batch[key] = arr.reshape(1, actual_len, 1) # 正确的目标形状

                elif key in ['obs', 'actions', 'rewards', 'dones', 'old_log_probs', 'actor_h_in', 'actor_c_in']:
                    # 处理包含 agent 维度的数据
                    if arr.ndim == 2: # 形状应为 (len, n_agents)
                        target_shape = (1, actual_len, args.n_agents)
                    elif arr.ndim == 3: # 形状应为 (len, n_agents, feature_dim)
                        target_shape = (1, actual_len, args.n_agents, -1)
                    else:
                        raise ValueError(f"Unexpected array dimension for key '{key}': {arr.ndim} with shape {arr.shape}")
                    batch[key] = arr.reshape(target_shape)

                elif key in ['state', 'next_state']:
                    # 处理全局状态 (无 agent 维度)
                    if arr.ndim != 2: # 形状应为 (len, state_dim)
                         raise ValueError(f"Unexpected array dimension for key '{key}': {arr.ndim} with shape {arr.shape}")
                    target_shape = (1, actual_len, -1)
                    batch[key] = arr.reshape(target_shape)
                else:
                    raise ValueError(f"Unknown key '{key}' in episode_data")

            # --- 调用学习方法 ---
            avg_actor_loss = 0
            avg_critic_loss = 0
            for agent_id in range(args.n_agents):
                 # 确保 batch 字典键存在且形状正确
                if batch['obs'].shape[1] > 0: # 检查序列长度是否大于0
                    actor_loss, critic_loss = agents[agent_id].learn(batch, shared_critic, shared_critic_optimizer)
                    actor_loss_history[agent_id].append(actor_loss)
                    if agent_id == 0:
                        critic_loss_history.append(critic_loss)
                        avg_critic_loss = critic_loss
                    avg_actor_loss += actor_loss
                else:
                    print(f"警告: 周期 {episode+1} 数据收集长度为 0，跳过学习步骤。")


            if args.n_agents > 0:
                avg_actor_loss /= args.n_agents

        # --- 日志记录 ---
        all_episode_rewards.append(episode_total_reward) # 存储当前周期的总奖励
        # 新增: 存储每个 agent 的原始周期奖励
        for ag_id in range(args.n_agents):
             raw_agent_rewards_per_episode[ag_id].append(episode_agent_rewards[ag_id])

        # 计算平滑平均奖励
        avg_reward = np.mean(all_episode_rewards[-100:]) # 总平均
        avg_rewards_history.append(avg_reward)

        # 新增: 计算每个 agent 的平滑平均奖励
        current_agent_avg_rewards = []
        for ag_id in range(args.n_agents):
            agent_avg = np.mean(raw_agent_rewards_per_episode[ag_id][-100:])
            agent_avg_rewards_history[ag_id].append(agent_avg)
            current_agent_avg_rewards.append(agent_avg)

        if (episode + 1) % args.log_interval == 0:
            # 更新打印信息
            agent_avgs_str = ", ".join([f"{avg:.2f}" for avg in current_agent_avg_rewards])
            print(f"Episode {episode + 1}/{args.n_episodes} | "
                  f"周期总奖励: {episode_total_reward:.2f} | "
                  f"平均总奖励(近100): {avg_reward:.2f} | "
                  f"各智能体平均奖励(近100): [{agent_avgs_str}] | "
                  f"Avg Actor Loss: {avg_actor_loss:.4f} | "
                  f"Critic Loss: {avg_critic_loss:.4f}")

    print("--- 训练结束 ---")

    # --- 保存结果和模型 ---
    np.save(os.path.join(args.save_dir, 'avg_total_rewards.npy'), np.array(avg_rewards_history))
    for i in range(args.n_agents):
         np.save(os.path.join(args.save_dir, f'agent_{i}_avg_rewards.npy'), np.array(agent_avg_rewards_history[i]))
         np.save(os.path.join(args.save_dir, f'actor_{i}_loss.npy'), np.array(actor_loss_history[i]))
         torch.save(agents[i].actor.state_dict(), os.path.join(args.save_dir, f'actor_{i}.pth'))
    np.save(os.path.join(args.save_dir, 'critic_loss.npy'), np.array(critic_loss_history))
    torch.save(shared_critic.state_dict(), os.path.join(args.save_dir, 'critic.pth'))
    print(f"训练结果和模型已保存到: {args.save_dir}")

    # --- 绘制结果 ---
    plt.figure(figsize=(12, 10)) # 调整画布大小

    # 更新奖励图
    plt.subplot(2, 1, 1) # 改为 2 行 1 列布局中的第一个
    plt.plot(avg_rewards_history, label='Average Total Reward (Smoothed)', linewidth=2, color='black')
    colors = plt.cm.viridis(np.linspace(0, 1, args.n_agents)) # 为智能体选择不同颜色
    for i in range(args.n_agents):
        plt.plot(agent_avg_rewards_history[i], label=f'Agent {i} Avg Reward (Smoothed)', alpha=0.8, color=colors[i])
    plt.title('Average Rewards per Episode (Smoothed over 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(loc='best') # 自动选择最佳图例位置
    plt.grid(True) # 添加网格线

    # 保持损失图 (放在第二行)
    plt.subplot(2, 2, 3) # 第二行第一个
    plt.plot(critic_loss_history)
    plt.title('Critic Loss')
    plt.xlabel('Training Step (Episode)')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 2, 4) # 第二行第二个
    for i in range(args.n_agents):
        plt.plot(actor_loss_history[i], label=f'Actor {i} Loss', color=colors[i])
    plt.title('Actor Losses')
    plt.xlabel('Training Step (Episode)')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)

    plt.tight_layout() # 调整布局防止重叠
    # 保存为新文件名
    save_path = os.path.join(args.save_dir, 'training_curves_detailed_rnn.png')
    plt.savefig(save_path)
    print(f"详细训练曲线图已保存到: {save_path}")
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAPPO (RNN) with Lead Time, SOP, & Reward Shaping (Omega)")

    # 环境参数
    parser.add_argument("--n_agents", type=int, default=3, help="Number of products/agents")
    parser.add_argument("--n_price_levels", type=int, default=5, help="Number of discrete price levels")
    parser.add_argument("--episode_len", type=int, default=200, help="Max steps per episode")

    # MAPPO 超参数
    parser.add_argument("--actor_lr", type=float, default=1e-5, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter (epsilon)")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--update_epochs", type=int, default=5, help="Number of PPO update epochs per learning step")
    parser.add_argument("--critic_hidden_dim", type=int, default=128, help="Hidden dimension for critic network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Hidden dimension for actor RNN layer")

    # 训练参数
    parser.add_argument("--n_episodes", type=int, default=30000, help="Total training episodes")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_dir", type=str, default="./results_rnn_inv_leadtime_lowhc_sop_omega", help="Directory for omega results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    # 新增: 控制同质化的标志
    parser.add_argument("--homogeneous", action='store_true', # 无需值，出现即为 True
                        help="If set, all agents will have the same cost, demand, and elasticity parameters.")

    # 添加冲击事件参数
    parser.add_argument("--shock_prob", type=float, default=0.05, help="Probability of a demand shock")
    parser.add_argument("--shock_duration", type=int, default=10, help="Duration of demand shocks")
    parser.add_argument("--shock_magnitude", type=float, default=0.6, help="Magnitude of demand shock")

    # 新增: 库存参数
    parser.add_argument("--initial_inventory", type=int, default=300, help="Initial inventory level")
    parser.add_argument("--holding_cost", type=float, default=0.01, help="Holding cost per unit")

    # 新增: 补货参数
    parser.add_argument("--replenishment_threshold", type=int, default=150, help="Inventory threshold")
    parser.add_argument("--replenishment_amount", type=int, default=400, help="Replenishment amount")
    parser.add_argument("--max_inventory_capacity", type=int, default=800, help="Maximum inventory capacity")

    # 新增: 补货提前期
    parser.add_argument("--replenishment_lead_time", type=int, default=10, help="Replenishment lead time")

    # 新增: 缺货惩罚参数
    parser.add_argument("--stockout_penalty", type=float, default=1.0, help="Penalty per unit of unsatisfied demand")

    # 新增: Omega 参数
    parser.add_argument("--omega", type=float, default=1.0, help="Weight for individual reward vs system reward (0 <= omega <= 1)")

    args = parser.parse_args()
    # 直接调用 main 函数
    main(args)
