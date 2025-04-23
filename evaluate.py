import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from environment import PricingEnvironment
from agent import MAPPOAgent # Actor is part of MAPPOAgent

def evaluate(args):
    """加载含库存管理的 RNN 模型并运行评估周期"""

    # --- 设置 ---
    device = torch.device(args.device)
    model_dir = args.model_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"从 '{model_dir}' 加载 RNN 模型 (含库存)，评估结果保存到 '{save_dir}'")
    print(f"使用设备: {device}")

    # --- 初始化环境 (确保传递所有参数) ---
    env = PricingEnvironment(
        n_products=args.n_agents,
        n_price_levels=args.n_price_levels,
        episode_length=args.episode_len,
        homogeneous=args.homogeneous, # 通常为 False 进行评估
        shock_prob=args.shock_prob,
        shock_duration=args.shock_duration,
        shock_magnitude=args.shock_magnitude,
        initial_inventory=args.initial_inventory,
        holding_cost=args.holding_cost,
        replenishment_threshold=args.replenishment_threshold, # <-- 添加
        replenishment_amount=args.replenishment_amount,   # <-- 添加
        max_inventory_capacity=args.max_inventory_capacity,
        replenishment_lead_time=args.replenishment_lead_time, # <-- 添加
        stockout_penalty_per_unit=args.stockout_penalty, # <-- 添加
        # omega 参数在评估环境时不需要
    )
    env_info = env.get_env_info()
    # 确认 obs_shape 是 4
    print("环境信息 (评估 RNN 含库存):", env_info)
    if env_info['obs_shape'] != 4:
        print(f"警告: 环境观察维度 ({env_info['obs_shape']}) 与预期 (4) 不符，请检查环境或参数！")

    # --- 初始化智能体 (使用新的 obs_dim) ---
    agents = []
    target_agent_id = args.target_agent_id # 获取要详细分析的目标 Agent ID
    if target_agent_id >= args.n_agents or target_agent_id < 0:
        print(f"错误: 目标 Agent ID {target_agent_id} 无效 (应在 0 到 {args.n_agents-1} 之间)")
        return

    for agent_id in range(args.n_agents):
        agent = MAPPOAgent(
            obs_dim=env_info['obs_shape'], # 使用环境提供的 obs_shape (应为 4)
            state_dim=env_info['state_shape'],
            action_dim=env_info['n_actions'],
            n_agents=args.n_agents,
            agent_id=agent_id,
            actor_lr=1e-5, critic_lr=5e-4, gamma=0.99, gae_lambda=0.95,
            clip_param=0.2, entropy_coef=0.01, update_epochs=1,
            rnn_hidden_dim=args.rnn_hidden_dim,
            device=device
        )
        actor_path = os.path.join(model_dir, f'actor_{agent_id}.pth')
        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
            agent.actor.eval()
            print(f"成功加载 RNN Actor {agent_id} 模型: {actor_path}")
        else:
            print(f"错误: 找不到 RNN Actor {agent_id} 模型: {actor_path}")
            return
        agents.append(agent)

    # --- 运行评估周期，记录更多信息 (包括库存) ---
    print(f"\n运行 {args.eval_episodes} 个评估周期...")
    if args.eval_episodes < 1: return

    # 存储第一个周期的详细数据
    eval_prices = []
    eval_obs_target = []      # 存储目标 Agent 的观察
    eval_sales_target = []    # 存储目标 Agent 的实际销售
    eval_inventory_target = [] # 存储目标 Agent 的库存 (步初)

    obs_list, _ = env.reset()
    actor_hidden_states = [agent.init_hidden() for agent in agents]
    episode_total_reward_eval = 0
    initial_inventory_target = env.current_inventory[target_agent_id]

    # 运行第一个评估周期
    for t in range(args.episode_len):
        current_actions = []
        temp_next_hidden_states = []

        # 记录目标 Agent 当前时间步的观察和库存 (步初)
        eval_obs_target.append(obs_list[target_agent_id])
        eval_inventory_target.append(env.current_inventory[target_agent_id])

        # 动作选择 (不变)
        with torch.no_grad():
            for agent_id in range(args.n_agents):
                action, _, next_hidden_state = agents[agent_id].choose_action(
                    obs_list[agent_id],
                    actor_hidden_states[agent_id],
                    deterministic=True
                )
                current_actions.append(action)
                temp_next_hidden_states.append(next_hidden_state)
        actor_hidden_states = temp_next_hidden_states

        eval_prices.append(current_actions)

        # 环境执行动作
        next_obs_list, _, rewards, dones, info = env.step(current_actions)

        # 记录目标 Agent 的实际销售量
        eval_sales_target.append(info['actual_sales'][target_agent_id])

        obs_list = next_obs_list
        episode_total_reward_eval += sum(rewards)

        if all(dones):
            # 在结束后记录最后一步的库存 (可选，因为前面记录的是步初库存)
            if args.n_agents > 2:
                 # eval_inventory_agent2.append(env.current_inventory[2]) # 如果想包含最后一步的期末库存
                 pass # 目前记录的是每步开始时的库存
            break

    print(f"评估周期 1 完成, 总奖励: {episode_total_reward_eval:.2f}, 步数: {t+1}")

    # --- 可视化所有智能体价格策略 ---
    eval_prices_arr = np.array(eval_prices) # 获取价格历史

    if eval_prices_arr.size > 0:
         # 调用新的/修改后的绘图函数来绘制所有智能体价格
         plot_all_agent_prices(eval_prices_arr,
                               args.n_agents,
                               args.n_price_levels,
                               args.save_dir)
    else:
         print("未能收集到足够的价格评估数据用于绘图。")

    # --- 可视化详细信息 ---
    eval_obs_target_arr = np.array(eval_obs_target)
    eval_sales_target_arr = np.array(eval_sales_target)
    eval_inventory_target_arr = np.array(eval_inventory_target)

    if (eval_prices_arr.size > 0 and eval_obs_target_arr.size > 0
        and eval_sales_target_arr.size > 0 and eval_inventory_target_arr.size > 0):
         if not os.path.exists(args.save_dir):
             print(f"警告: 保存目录 '{args.save_dir}' 不存在，将尝试创建。")
             try:
                 os.makedirs(args.save_dir)
             except OSError as e:
                 print(f"错误: 无法创建目录 '{args.save_dir}': {e}")
                 return

         # 调用详细绘图函数，传入 omega 值
         plot_agent_details_inv(eval_prices_arr, eval_obs_target_arr,
                               eval_sales_target_arr, eval_inventory_target_arr,
                               target_agent_id=args.target_agent_id,
                               n_price_levels=args.n_price_levels,
                               initial_inventory=args.initial_inventory,
                               save_dir=args.save_dir,
                               max_capacity=args.max_inventory_capacity,
                               episode_length=args.episode_len,
                               omega_value=args.omega) # <-- 传入 omega 值给绘图函数
    else:
         print("未能收集到足够的评估数据用于详细绘图。")


def plot_all_agent_prices(prices_history, n_agents, n_price_levels, save_dir):
    """
    绘制评估周期中所有智能体的价格等级选择。
    """
    plt.figure(figsize=(12, 6))
    episode_length = prices_history.shape[0]
    timesteps = np.arange(episode_length)
    colors = plt.cm.viridis(np.linspace(0, 1, n_agents)) # 颜色映射

    for i in range(n_agents):
        # 绘制每个智能体的价格等级
        plt.plot(timesteps, prices_history[:, i], label=f'Agent {i} Price Level', color=colors[i], marker='.', linestyle='-')

    plt.title('Learned Pricing Strategy (All Agents, RNN with Inventory)') # 更新标题
    plt.xlabel('Time Step in Episode')
    plt.ylabel('Price Level (0 = Lowest, {} = Highest)'.format(n_price_levels - 1))
    plt.yticks(np.arange(n_price_levels))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    filename = "evaluation_all_prices_rnn_inv.png" # 新文件名
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"所有智能体价格策略图已保存到: {save_path}")
    # plt.show()


def plot_agent_details_inv(prices_history, obs_history, sales_history, inventory_history,
                           target_agent_id, n_price_levels, initial_inventory, save_dir,
                           max_capacity, episode_length, omega_value):
    """
    绘制目标智能体的详细评估信息：价格、观察、销售、库存。
    """
    n_agents = prices_history.shape[1]
    if obs_history.shape[0] != episode_length or sales_history.shape[0] != episode_length or inventory_history.shape[0] != episode_length:
        print(f"错误：数据长度不匹配！ Price: {episode_length}, Obs: {obs_history.shape[0]}, Sales: {sales_history.shape[0]}, Inv: {inventory_history.shape[0]}")
        return

    timesteps = np.arange(episode_length)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True) # 4 个子图

    # --- Panel 1: Agent Price Level (不变) ---
    axes[0].plot(timesteps, prices_history[:, target_agent_id], label=f'Agent {target_agent_id} Price Level', color='red', marker='.', linestyle='-')
    axes[0].set_title(f'Agent {target_agent_id} Detailed Eval (LeadTime, LowHC, SOP, Omega={omega_value}, {episode_length} steps)') # 更新标题
    axes[0].set_ylabel('Price Level')
    axes[0].set_yticks(np.arange(n_price_levels))
    axes[0].grid(True)
    axes[0].legend()

    # --- Panel 2: Agent Observations (增加库存观察) ---
    if obs_history.shape[1] >= 4:
        axes[1].plot(timesteps, obs_history[:, 0], label='Norm. Own Price', color='blue', linestyle='--')
        axes[1].plot(timesteps, obs_history[:, 1], label='Norm. Time', color='grey', linestyle=':')
        axes[1].plot(timesteps, obs_history[:, 2], label='Norm. Avg Comp. Price', color='green', linestyle='-.')
        axes[1].plot(timesteps, obs_history[:, 3], label='Norm. Inventory', color='orange', linestyle='-') # 新增
    else:
        axes[1].text(0.5, 0.5, 'Observation data dimension incorrect', horizontalalignment='center', verticalalignment='center')
    axes[1].set_ylabel('Normalized Observation')
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_ylim([-0.1, 1.1])

    # --- Panel 3: Agent Actual Sales ---
    axes[2].plot(timesteps, sales_history, label=f'Agent {target_agent_id} Actual Sales', color='purple', linestyle='-')
    axes[2].set_ylabel('Actual Sales')
    axes[2].grid(True)
    axes[2].legend()
    min_sales = np.min(sales_history)
    max_sales = np.max(sales_history)
    axes[2].set_ylim([min_sales - 2, max_sales + 2]) # 调整 Y 轴范围

    # --- Panel 4: Agent Inventory Level ---
    axes[3].plot(timesteps, inventory_history, label=f'Agent {target_agent_id} Inventory (Start of Step)', color='brown', linestyle='-')
    axes[3].axhline(0, color='black', linestyle='--', linewidth=0.8, label='Zero Inventory') # 参考线
    axes[3].set_xlabel(f'Time Step in Episode (Total Steps: {episode_length})')
    axes[3].set_ylabel('Inventory Level')
    axes[3].grid(True)
    axes[3].legend()
    axes[3].set_ylim([-5, max_capacity + 10]) # 使用传入的最大容量

    plt.tight_layout()

    # 修改: 更新文件名生成逻辑，包含 omega 值
    filename = f"evaluation_agent{target_agent_id}_details_omega{omega_value}_{episode_length}steps.png"
    save_path = os.path.join(save_dir, filename)
    print(f"准备保存评估图到: {os.path.abspath(save_path)}")
    try:
        plt.savefig(save_path)
        print(f"Agent {target_agent_id} 详细评估图 (Omega={omega_value}, {episode_length} 步) 已成功保存。")
    except Exception as e:
        print(f"错误: 保存评估图时出错: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MAPPO (RNN) with Lead Time, SOP, & Reward Shaping (Omega)")

    # 环境和模型参数
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--n_price_levels", type=int, default=5, help="Number of price levels")
    parser.add_argument("--episode_len", type=int, default=100, help="Episode length for evaluation")
    parser.add_argument("--homogeneous", type=bool, default=False, help="Whether the agents are homogeneous (should match training, typically False for eval)")
    # 冲击参数 (应匹配训练)
    parser.add_argument("--shock_prob", type=float, default=0.05, help="Shock probability")
    parser.add_argument("--shock_duration", type=int, default=10, help="Shock duration")
    parser.add_argument("--shock_magnitude", type=float, default=0.6, help="Shock magnitude")
    # 库存参数 (应匹配训练)
    parser.add_argument("--initial_inventory", type=int, default=300, help="Initial inventory level")
    parser.add_argument("--holding_cost", type=float, default=0.1, help="Holding cost per unit")
    # 模型参数
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Actor RNN hidden dim")
    parser.add_argument("--critic_hidden_dim", type=int, default=128, help="Critic hidden dim")

    # 评估特定参数
    parser.add_argument("--model_dir", type=str, default="./results_rnn_inv_leadtime_lowhc_sop_omega08", help="Directory containing trained models (specific omega run)")
    parser.add_argument("--save_dir", type=str, default="./results_rnn_inv_leadtime_lowhc_sop_omega08", help="Directory to save evaluation results")
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes (only first is detailed)")
    parser.add_argument("--target_agent_id", type=int, default=1, help="ID of the agent for detailed plotting")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--max_inventory_capacity", type=int, default=800, help="Maximum inventory capacity per agent")
    parser.add_argument("--omega", type=float, default=1.0, help="Omega value used during training (for plotting info)")
    parser.add_argument("--replenishment_threshold", type=int, default=150, help="Inventory threshold (match training)")
    parser.add_argument("--replenishment_amount", type=int, default=400, help="Replenishment amount (match training)")
    parser.add_argument("--replenishment_lead_time", type=int, default=10, help="Replenishment lead time (match training)")
    parser.add_argument("--stockout_penalty", type=float, default=1.0, help="Stockout penalty (match training)")

    args = parser.parse_args()
    print(f"--- 开始评估 (Agent: {args.target_agent_id}, Len: {args.episode_len}, Omega={args.omega} Model) ---")
    print(f"模型加载自: {args.model_dir}")
    print(f"评估结果将保存到: {args.save_dir}")
    evaluate(args)
