import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse # 用于命令行参数
import random # 导入 random

from environment import PricingEnvironment
from agent import MAPPOAgent, Critic # 导入 MAPPOAgent 和 Critic

def main(args):
    """主训练函数 (RNN 版本)"""

    # --- 设置随机种子 ---
    seed = 42 # 或者其他固定值
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"使用固定随机种子: {seed}")

    # --- 设置 ---
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 检查 omega 范围 (保留此检查)
    if not (0.0 <= args.omega <= 1.0):
         print(f"警告: Omega 值 ({args.omega}) 不在 [0, 1] 范围内，将强制设为 1.0 (纯个体奖励)")
         args.omega = 1.0

    # 新增: 根据 stockout_penalty 和 n_agents 构建保存目录
    # 去掉末尾可能存在的斜杠，确保拼接正确
    base_save_dir = args.save_dir.rstrip('/\\\\')
    # 修改: 使用 n_agents 和 stockout_penalty 构建目录名
    args.save_dir = f"{base_save_dir}_N{args.n_agents}_penalty_{args.stockout_penalty:.1f}"

    # 创建结果保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f"结果将保存到: {args.save_dir}") # 打印最终的保存路径

    # --- 初始化环境 (传入所有参数) ---
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
    # 新增: 记录每个周期的原始系统总奖励
    raw_original_system_rewards_history = []
    # 修改: raw_agent_rewards_per_episode 用于存储 *原始* 奖励
    raw_original_agent_rewards_per_episode = [[] for _ in range(args.n_agents)]
    # 保持: 记录每个智能体的平滑 *塑形后* 奖励历史 (用于原有曲线图)
    agent_avg_shaped_rewards_history = [[] for _ in range(args.n_agents)]
    # 新增: 记录每个智能体的平滑 *原始* 奖励历史
    agent_avg_original_rewards_history = [[] for _ in range(args.n_agents)]

    print("\n--- 开始训练 (RNN Actor) ---")
    for episode in range(args.n_episodes):
        # -- 数据收集 (需要存储隐藏状态) --
        episode_data = {
            'obs': [], 'state': [], 'actions': [], 'rewards': [],
            'dones': [], 'old_log_probs': [], 'next_state': [],
            'actor_h_in': [], 'actor_c_in': [], # 存储 Actor 的 LSTM 输入隐藏状态
            'masks': [] # 用于标记有效步骤
        }
        episode_total_shaped_reward = 0
        # 修改: step_rewards 现在是 numpy array，便于累加
        episode_agent_shaped_rewards = np.zeros(args.n_agents) # 记录本周期每个 agent 的总奖励
        # 新增: 累加 *原始* 系统奖励
        current_episode_original_system_reward = 0
        # 新增: 累加 *原始* 个体奖励
        current_episode_original_agent_rewards = np.zeros(args.n_agents)

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
            # 修改: 累加每个 agent 的 *塑形后* 奖励用于日志和原有曲线图
            episode_agent_shaped_rewards += np.array(rewards)
            # 新增: 累加 *原始* 系统奖励
            current_episode_original_system_reward += sum(info['original_rewards'])
            # 新增: 累加 *原始* 个体奖励
            current_episode_original_agent_rewards += np.array(info['original_rewards'])

            if all(dones):
                break

        # 周期结束，计算总奖励
        episode_total_shaped_reward = episode_agent_shaped_rewards.sum()

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
        all_episode_rewards.append(episode_total_shaped_reward) # 存储当前周期的总 *塑形后* 奖励
        # 新增: 存储当前周期的 *原始* 系统总奖励
        raw_original_system_rewards_history.append(current_episode_original_system_reward)
        # 修改: 存储每个 agent 的 *原始* 周期奖励
        for ag_id in range(args.n_agents):
             raw_original_agent_rewards_per_episode[ag_id].append(current_episode_original_agent_rewards[ag_id])

        # 计算平滑平均 *塑形后* 奖励 (用于原有图和日志)
        avg_shaped_reward = np.mean(all_episode_rewards[-100:]) # 总平均 (塑形后)
        avg_rewards_history.append(avg_shaped_reward) # 历史记录也是塑形后的

        # 修改: 计算每个 agent 的平滑平均 *塑形后* 奖励
        current_agent_avg_shaped_rewards = []
        for ag_id in range(args.n_agents):
            # -- 确保旧的、错误的计算方式被注释掉或删除 --
            # agent_avg_shaped = np.mean([raw_rewards_list[-1] for raw_rewards_list in raw_original_agent_rewards_per_episode[ag_id][-100:]]) # <--- 确认此行被注释或删除

            # -- 正确计算平滑 *原始* 奖励 --
            if len(raw_original_agent_rewards_per_episode[ag_id]) > 0:
                 # 确认这部分是正确的列表推导式
                 # 添加类型检查以防万一
                 valid_rewards = [reward for reward in raw_original_agent_rewards_per_episode[ag_id][-100:] if isinstance(reward, (int, float, np.number))]
                 if valid_rewards: # 确保列表非空
                    agent_avg_original = np.mean(valid_rewards)
                 else:
                    agent_avg_original = 0.0 # 如果过滤后为空
            else:
                 agent_avg_original = 0.0 # 如果还没有数据
            agent_avg_original_rewards_history[ag_id].append(agent_avg_original)

            # -- 保留下面的占位符逻辑用于原有的塑形奖励图 (这部分看起来没问题) --
            agent_avg_placeholder = avg_shaped_reward / args.n_agents
            agent_avg_shaped_rewards_history[ag_id].append(agent_avg_placeholder)
            current_agent_avg_shaped_rewards.append(agent_avg_placeholder)

        if (episode + 1) % args.log_interval == 0:
            # 更新打印信息 (使用占位符和新的原始平均值)
            # agent_avgs_str = ", ".join([f"{avg:.2f}" for avg in current_agent_avg_shaped_rewards]) # 旧的占位符格式
            current_agent_avg_originals = [hist[-1] if hist else 0 for hist in agent_avg_original_rewards_history]
            agent_orig_avgs_str = ", ".join([f"{avg:.2f}" for avg in current_agent_avg_originals])
            print(f"Episode {episode + 1}/{args.n_episodes} | "
                  f"周期总奖励(塑形): {episode_total_shaped_reward:.2f} | "
                  f"平均总奖励(塑形,近100): {avg_shaped_reward:.2f} | "
                  f"原始总奖励: {current_episode_original_system_reward:.2f} | "
                  f"各智能体平均原始奖励(近100): [{agent_orig_avgs_str}] | " # 添加原始平均奖励
                  f"Avg Actor Loss: {avg_actor_loss:.4f} | "
                  f"Critic Loss: {avg_critic_loss:.4f}")

    print("--- 训练结束 ---")

    # --- 保存结果和模型 ---
    np.save(os.path.join(args.save_dir, 'avg_total_rewards.npy'), np.array(avg_rewards_history))
    for i in range(args.n_agents):
         np.save(os.path.join(args.save_dir, f'agent_{i}_avg_rewards.npy'), np.array(agent_avg_shaped_rewards_history[i]))
         np.save(os.path.join(args.save_dir, f'actor_{i}_loss.npy'), np.array(actor_loss_history[i]))
         torch.save(agents[i].actor.state_dict(), os.path.join(args.save_dir, f'actor_{i}.pth'))
    np.save(os.path.join(args.save_dir, 'critic_loss.npy'), np.array(critic_loss_history))
    torch.save(shared_critic.state_dict(), os.path.join(args.save_dir, 'critic.pth'))
    print(f"训练结果和模型已保存到: {args.save_dir}")

    # --- 创建 BASELINE 目录 ---
    baseline_dir = os.path.join(args.save_dir, "BASELINE")
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
        print(f"创建 Baseline 绘图目录: {baseline_dir}")

    # --- 辅助函数：计算移动平均 ---
    def moving_average(data, window_size=100):
        if len(data) < window_size:
            print(f"警告: 数据长度 {len(data)} 小于窗口大小 {window_size}，无法计算有效的移动平均。")
            return np.array([]) # 数据太少，无法计算有效移动平均
        # 使用 valid 模式确保输出长度正确
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # --- 绘制新的原始奖励图 ---
    print("--- 生成原始奖励对比图 (保存至 BASELINE 目录) ---")

    # 1. 绘制系统原始奖励 (Raw vs Smoothed)
    try:
        plt.figure(figsize=(12, 6))
        raw_sys_rewards = np.array(raw_original_system_rewards_history)
        smoothed_sys_rewards = moving_average(raw_sys_rewards)
        episodes_raw = np.arange(len(raw_sys_rewards))
        # 移动平均后的数据点数会减少，需要调整 x 轴
        # (窗口大小 - 1) / 2 用于居中对齐，或者直接从 window_size - 1 开始
        episodes_smoothed = np.arange(len(smoothed_sys_rewards)) + (100 - 1)

        plt.plot(episodes_raw, raw_sys_rewards, label='Raw Original System Reward', alpha=0.4, color='lightblue')
        if smoothed_sys_rewards.size > 0:
            plt.plot(episodes_smoothed, smoothed_sys_rewards, label='Smoothed Original System Reward (window=100)', color='blue', linewidth=2)
        else:
             print("系统原始奖励数据不足，无法绘制平滑曲线。")
        plt.title('Original System Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Original System Reward')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        sys_plot_path = os.path.join(baseline_dir, "system_original_reward_raw_vs_smoothed.png")
        plt.savefig(sys_plot_path)
        print(f"系统原始奖励图已保存到: {sys_plot_path}")
        plt.close() # 关闭图形，避免干扰后续绘图
    except Exception as e:
        print(f"错误: 绘制系统原始奖励图时出错: {e}")

    # 2. 绘制各智能体原始奖励 (Raw vs Smoothed)
    for i in range(args.n_agents):
        try:
            plt.figure(figsize=(12, 6))
            raw_agent_rewards = np.array(raw_original_agent_rewards_per_episode[i])
            smoothed_agent_rewards = moving_average(raw_agent_rewards)
            episodes_raw_agent = np.arange(len(raw_agent_rewards))
            episodes_smoothed_agent = np.arange(len(smoothed_agent_rewards)) + (100 - 1)

            plt.plot(episodes_raw_agent, raw_agent_rewards, label=f'Agent {i} Raw Original Reward', alpha=0.4, color='lightgreen')
            if smoothed_agent_rewards.size > 0:
                plt.plot(episodes_smoothed_agent, smoothed_agent_rewards, label=f'Agent {i} Smoothed Original Reward (window=100)', color='green', linewidth=2)
            else:
                 print(f"智能体 {i} 原始奖励数据不足，无法绘制平滑曲线。")
            plt.title(f'Agent {i} Original Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Individual Original Reward')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            agent_plot_path = os.path.join(baseline_dir, f"agent_{i}_original_reward_raw_vs_smoothed.png")
            plt.savefig(agent_plot_path)
            print(f"智能体 {i} 原始奖励图已保存到: {agent_plot_path}")
            plt.close() # 关闭图形
        except Exception as e:
             print(f"错误: 绘制智能体 {i} 原始奖励图时出错: {e}")

    # --- 绘制原有的结果 (损失和塑形后奖励) ---
    print("--- 生成原有训练曲线图 (损失和塑形后奖励) ---")
    plt.figure(figsize=(12, 10)) # 调整画布大小

    # 更新奖励图
    plt.subplot(2, 1, 1) # 改为 2 行 1 列布局中的第一个
    plt.plot(avg_rewards_history, label='Average Total Reward (Smoothed)', linewidth=2, color='black')
    colors = plt.cm.viridis(np.linspace(0, 1, args.n_agents)) # 为智能体选择不同颜色
    for i in range(args.n_agents):
        plt.plot(agent_avg_shaped_rewards_history[i], label=f'Agent {i} Avg Reward (Smoothed)', alpha=0.8, color=colors[i])
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
    parser.add_argument("--n_episodes", type=int, default=10000, help="Total training episodes")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_dir", type=str, default="./results_exp5_agent_scalability", help="Base directory for experiment 5 results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    # 新增: 控制同质化的标志
    parser.add_argument("--homogeneous", action='store_true', # 无需值，出现即为 True
                        help="If set, all agents will have the same cost, demand, and elasticity parameters.")

    # 添加冲击事件参数
    parser.add_argument("--shock_prob", type=float, default=0.05, help="Probability of a demand shock")
    parser.add_argument("--shock_duration", type=int, default=10, help="Duration of demand shocks")
    parser.add_argument("--shock_magnitude", type=float, default=0.6, help="Magnitude of demand shock")

    # 新增: 库存参数
    parser.add_argument("--initial_inventory", type=int, default=500, help="Initial inventory level")
    parser.add_argument("--holding_cost", type=float, default=0.1, help="Holding cost per unit")

    # 新增: 补货参数
    parser.add_argument("--replenishment_threshold", type=int, default=200, help="Inventory threshold")
    parser.add_argument("--replenishment_amount", type=int, default=100, help="Replenishment amount")
    parser.add_argument("--max_inventory_capacity", type=int, default=800, help="Maximum inventory capacity")

    # 新增: 补货提前期
    parser.add_argument("--replenishment_lead_time", type=int, default=5, help="Replenishment lead time")

    # 新增: 缺货惩罚参数
    parser.add_argument("--stockout_penalty", type=float, default=0.01, help="Penalty per unit of unsatisfied demand")

    # 新增: Omega 参数
    parser.add_argument("--omega", type=float, default=1.0, help="Weight for individual reward vs system reward (0 <= omega <= 1)")

    args = parser.parse_args()

    # 确保在调用 main 前检查 omega 并设置 save_dir
    # (已将此逻辑移入 main 函数开头)

    # 直接调用 main 函数
    main(args)
