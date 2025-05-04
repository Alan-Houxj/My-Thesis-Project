import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns # 用于绘制热力图
import random # 导入 random

from environment import PricingEnvironment
from agent import MAPPOAgent # Actor is part of MAPPOAgent

def run_baseline_episode(env, policy_type, episode_len, n_agents, n_price_levels):
    """Runs one episode using a fixed baseline policy.

    Args:
        env: The PricingEnvironment instance.
        policy_type (str): 'fixed_low', 'fixed_high', or 'random'.
        episode_len (int): Max steps for the episode.
        n_agents (int): Number of agents.
        n_price_levels (int): Number of price levels.

    Returns:
        float: Total original system reward for this episode.
    """
    obs_list, state = env.reset()
    done = False
    episode_original_reward = 0.0
    step = 0

    while not done and step < episode_len:
        if policy_type == 'fixed_low':
            actions = [0] * n_agents # Lowest price level index is 0
        elif policy_type == 'fixed_high':
            actions = [n_price_levels - 1] * n_agents # Highest price level index
        elif policy_type == 'random':
            actions = np.random.randint(0, n_price_levels, n_agents)
        else:
            raise ValueError(f"Unknown baseline policy type: {policy_type}")

        next_obs_list, next_state, shaped_rewards, dones, info = env.step(actions)

        # IMPORTANT: Accumulate ORIGINAL system reward
        episode_original_reward += sum(info['original_rewards'])

        obs_list = next_obs_list
        state = next_state
        done = all(dones)
        step += 1

    return episode_original_reward

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
                           max_capacity, episode_length):
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
    axes[0].set_title(f'Agent {target_agent_id} Detailed Evaluation ({episode_length} steps)')
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

    # 修改: 更新文件名生成逻辑，移除 omega 值
    filename = f"evaluation_agent{target_agent_id}_details_{episode_length}steps.png"
    save_path = os.path.join(save_dir, filename)
    print(f"准备保存评估图到: {os.path.abspath(save_path)}")
    try:
        plt.savefig(save_path)
        print(f"Agent {target_agent_id} 详细评估图 ({episode_length} 步) 已成功保存。")
    except Exception as e:
        print(f"错误: 保存评估图时出错: {e}")

# --- 新增：绘制价格选择分布热力图 ---
def plot_price_distribution_heatmap(prices_history, n_agents, n_price_levels, save_dir):
    """
    绘制每个智能体选择各价格等级的频率热力图。
    """
    try:
        episode_len, _ = prices_history.shape
        price_counts = np.zeros((n_agents, n_price_levels), dtype=int)

        for agent_id in range(n_agents):
            for price_level in range(n_price_levels):
                price_counts[agent_id, price_level] = np.sum(prices_history[:, agent_id] == price_level)

        # 转换为百分比 (可选，但更直观)
        price_percentages = (price_counts / episode_len) * 100

        plt.figure(figsize=(8, max(4, n_agents * 0.8))) # 调整高度以适应智能体数量
        sns.heatmap(price_percentages, annot=True, fmt=".1f", cmap="viridis",
                    xticklabels=np.arange(n_price_levels),
                    yticklabels=[f"Agent {i}" for i in range(n_agents)],
                    cbar_kws={'label': 'Selection Percentage (%)'})
        plt.xlabel("Price Level")
        plt.ylabel("Agent ID")
        plt.title("Price Level Selection Distribution")
        plt.tight_layout()

        filename = "evaluation_price_distribution_heatmap.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"价格分布热力图已保存到: {save_path}")
        plt.close()
    except Exception as e:
        print(f"错误: 绘制价格分布热力图时出错: {e}")

# --- 新增：绘制价格协调度 (标准差) 图 ---
def plot_price_coordination(prices_history, save_dir):
    """
    绘制价格协调度 (跨智能体价格标准差) 随时间变化的图。
    """
    try:
        episode_len, n_agents = prices_history.shape
        if n_agents < 2:
            print("只有一个智能体，无法计算价格协调度。")
            return

        timesteps = np.arange(episode_len)
        # 计算每个时间步的价格标准差
        price_std_dev = np.std(prices_history, axis=1)

        plt.figure(figsize=(12, 5))
        plt.plot(timesteps, price_std_dev, label='Price Std Dev across Agents', color='orange')
        plt.xlabel("Time Step in Episode")
        plt.ylabel("Price Level Standard Deviation")
        plt.title("Price Coordination Over Time")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0) # 标准差不会是负数
        plt.tight_layout()

        filename = "evaluation_price_coordination.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"价格协调度图已保存到: {save_path}")
        plt.close()
    except Exception as e:
        print(f"错误: 绘制价格协调度图时出错: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MAPPO (RNN) vs Baselines (Exp 1)") # 更新描述

    # --- 环境和模型参数 (与 main.py 对齐) ---
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--n_price_levels", type=int, default=5, help="Number of price levels")
    parser.add_argument("--episode_len", type=int, default=200, help="Max steps per episode") # 增加默认评估长度
    parser.add_argument("--homogeneous", type=bool, default=False, help="Whether agents are homogeneous")
    parser.add_argument("--shock_prob", type=float, default=0.05, help="Shock probability")
    parser.add_argument("--shock_duration", type=int, default=10, help="Shock duration")
    parser.add_argument("--shock_magnitude", type=float, default=0.6, help="Shock magnitude")
    parser.add_argument("--initial_inventory", type=int, default=500, help="Initial inventory")
    parser.add_argument("--holding_cost", type=float, default=0.1, help="Holding cost")
    parser.add_argument("--replenishment_threshold", type=int, default=200, help="Replenishment threshold")
    parser.add_argument("--replenishment_amount", type=int, default=100, help="Replenishment amount")
    parser.add_argument("--max_inventory_capacity", type=int, default=800, help="Max inventory capacity")
    parser.add_argument("--replenishment_lead_time", type=int, default=5, help="Replenishment lead time")
    parser.add_argument("--stockout_penalty", type=float, default=0.01, help="Stockout penalty")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Actor RNN hidden dim")
    # parser.add_argument("--critic_hidden_dim", type=int, default=128, help="Critic hidden dim") # Critic 不在此脚本加载
    parser.add_argument("--omega", type=float, default=0.8, help="Omega value used during training (for info/plotting)") # 保留用于绘图

    # --- 评估特定参数 ---
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained MARL models") # 改为必须指定
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save evaluation results/plots (defaults to model_dir)") # 默认改为 None
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes for averaging results") # 增加默认评估周期数
    parser.add_argument("--target_agent_id", type=int, default=1, help="ID of agent for detailed MARL plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

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
    model_dir = args.model_dir
    # 修改: 如果 save_dir 未指定，则默认为 model_dir
    save_dir = args.save_dir if args.save_dir is not None else model_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"--- 开始评估 (实验二: MAPPO 策略评估) ---") # 更新标题
    print(f"模型加载自: {model_dir}")
    print(f"评估结果将保存到: {save_dir}")
    print(f"运行 {args.eval_episodes} 个评估周期进行平均。")
    print(f"使用设备: {device}")

    # --- 初始化环境 (只需一次) ---
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
        omega=args.omega # Omega 仍然传递，环境内部可能需要它
    )
    env_info = env.get_env_info()

    # --- 评估 MAPPO-RNN 模型 ---
    print("\n--- 评估 MAPPO-RNN 模型 ---")
    marl_agents = []
    try:
        for agent_id in range(args.n_agents):
            agent = MAPPOAgent(
                obs_dim=env_info['obs_shape'], state_dim=env_info['state_shape'],
                action_dim=env_info['n_actions'], n_agents=args.n_agents, agent_id=agent_id,
                rnn_hidden_dim=args.rnn_hidden_dim, device=device # 其他超参数在 Agent 内部有默认值，加载模型会覆盖网络
            )
            actor_path = os.path.join(model_dir, f'actor_{agent_id}.pth')
            if os.path.exists(actor_path):
                agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
                agent.actor.eval()
                print(f"成功加载 Actor {agent_id} 模型: {actor_path}")
            else:
                raise FileNotFoundError(f"找不到 Actor {agent_id} 模型: {actor_path}")
            marl_agents.append(agent)
    except Exception as e:
        print(f"加载 MARL 模型失败: {e}")
        exit()

    marl_total_rewards = [] # 记录每个周期的系统总原始奖励
    marl_individual_rewards = [[] for _ in range(args.n_agents)] # 记录每个agent每个周期的原始奖励
    # 新增: 记录塑形奖励
    marl_total_shaped_rewards = [] # 记录每个周期的系统总塑形奖励
    marl_individual_shaped_rewards = [[] for _ in range(args.n_agents)] # 记录每个agent每个周期的塑形奖励
    # 新增: 记录每个周期的平均价格波动性
    marl_episode_avg_volatility = []
    # 新增: 实验三指标累加器
    total_inventory_sum = 0.0
    total_stockout_event_count = 0
    total_agent_steps = 0
    # 存储第一个周期的详细数据用于绘图
    first_episode_data = {'prices': [], 'obs_target': [], 'sales_target': [], 'inventory_target': []}

    for ep in range(args.eval_episodes):
        obs_list, state = env.reset()
        actor_hidden_states = [agent.init_hidden() for agent in marl_agents]
        episode_original_reward = 0.0
        episode_individual_original_rewards = np.zeros(args.n_agents)
        # 新增: 初始化周期塑形奖励累加器
        episode_shaped_reward = 0.0
        episode_individual_shaped_rewards = np.zeros(args.n_agents)
        # 新增: 记录当前周期的价格历史
        current_episode_prices = []
        done = False
        step = 0

        # 记录第一个周期的详细数据
        is_first_episode = (ep == 0)
        if is_first_episode:
            # 记录第一个周期的初始库存
            if args.target_agent_id < args.n_agents:
                 first_episode_data['inventory_target'].append(env.current_inventory[args.target_agent_id])
            else:
                 first_episode_data['inventory_target'].append(0) # Handle invalid target id

        while not done and step < args.episode_len:
            current_actions = []
            temp_next_hidden_states = []

            # 记录第一个周期的观察
            if is_first_episode:
                if args.target_agent_id < args.n_agents:
                    first_episode_data['obs_target'].append(obs_list[args.target_agent_id])
                else:
                    # Append zeros or handle appropriately if target_id is invalid
                    first_episode_data['obs_target'].append(np.zeros(env_info['obs_shape']))

            with torch.no_grad():
                for agent_id in range(args.n_agents):
                    action, _, next_hidden_state = marl_agents[agent_id].choose_action(
                        obs_list[agent_id], actor_hidden_states[agent_id], deterministic=True
                    )
                    current_actions.append(action)
                    temp_next_hidden_states.append(next_hidden_state)
            actor_hidden_states = temp_next_hidden_states

            # 新增: 记录当前时间步的价格
            current_episode_prices.append(current_actions)

            next_obs_list, next_state, shaped_rewards, dones, info = env.step(current_actions)

            # 累加原始系统奖励
            episode_original_reward += sum(info['original_rewards'])
            # 累加原始个体奖励
            episode_individual_original_rewards += np.array(info['original_rewards'])
            # 新增: 累加塑形奖励
            episode_individual_shaped_rewards += np.array(shaped_rewards)

            # 新增: 累加实验三指标
            total_inventory_sum += sum(info['inventory'])
            total_stockout_event_count += sum(1 for demand in info['unsatisfied_demand'] if demand > 0)
            total_agent_steps += args.n_agents

            # 记录第一个周期的价格 (现在从 current_episode_prices 获取)
            # if is_first_episode:
            #     first_episode_data['prices'].append(current_actions)

            # 记录第一个周期的销售和库存 (步末，即下步步初)
            if is_first_episode:
                if args.target_agent_id < args.n_agents:
                    first_episode_data['sales_target'].append(info['actual_sales'][args.target_agent_id])
                    # 记录下步步初库存
                    if not all(dones):
                        first_episode_data['inventory_target'].append(env.current_inventory[args.target_agent_id])
                else:
                    first_episode_data['sales_target'].append(0)
                    if not all(dones):
                         first_episode_data['inventory_target'].append(0)

            obs_list = next_obs_list
            state = next_state
            done = all(dones)
            step += 1
        # --- 周期结束 ---

        # 存储系统总原始奖励
        marl_total_rewards.append(episode_original_reward)
        # 存储个体原始奖励
        for agent_id in range(args.n_agents):
            marl_individual_rewards[agent_id].append(episode_individual_original_rewards[agent_id])

        # 新增: 存储塑形奖励
        # marl_total_shaped_rewards.append(episode_shaped_reward) # <--- 移除存储错误累加值的代码
        # 直接存储正确的个体塑形奖励
        for agent_id in range(args.n_agents):
            # 注意：这里需要的是周期的总个体塑形奖励，而不是列表
            # episode_individual_shaped_rewards 已经是周期的总和了
            marl_individual_shaped_rewards[agent_id].append(episode_individual_shaped_rewards[agent_id])

        # 新增: 计算并存储本周期的平均价格波动性
        if current_episode_prices: # Ensure not empty
            prices_arr = np.array(current_episode_prices)
            if prices_arr.shape[1] > 1: # Need at least 2 agents to calc std dev
                timestep_std_dev = np.std(prices_arr, axis=1)
                avg_volatility = np.mean(timestep_std_dev)
                marl_episode_avg_volatility.append(avg_volatility)
                # 如果是第一个周期，也填充 first_episode_data['prices']
                if is_first_episode:
                    first_episode_data['prices'] = current_episode_prices # 列表形式存储
            else:
                 marl_episode_avg_volatility.append(0.0) # Volatility is 0 for single agent
                 if is_first_episode:
                     first_episode_data['prices'] = current_episode_prices
        else:
             marl_episode_avg_volatility.append(0.0) # Append 0 if no steps occurred
             if is_first_episode:
                 first_episode_data['prices'] = []

        # 打印 MARL 评估进度
        if (ep + 1) % max(1, args.eval_episodes // 10) == 0 or args.eval_episodes <= 10:
             print(f"  MAPPO Episode {ep+1}/{args.eval_episodes} 完成, 原始奖励: {episode_original_reward:.2f}")

    # --- 计算平均结果 ---
    avg_marl_reward = np.mean(marl_total_rewards) if marl_total_rewards else 0
    avg_marl_individual_rewards = [np.mean(rewards) if rewards else 0 for rewards in marl_individual_rewards]
    # 新增: 计算平均塑形奖励
    # avg_marl_shaped_reward = np.mean(marl_total_shaped_rewards) if marl_total_shaped_rewards else 0 # <--- 不再需要这个列表
    avg_marl_shaped_reward = avg_marl_reward # 理论上总塑形奖励等于总原始奖励
    avg_marl_individual_shaped_rewards = [np.mean(rewards) if rewards else 0 for rewards in marl_individual_shaped_rewards]
    # 新增: 计算平均价格波动性
    avg_marl_volatility = np.mean(marl_episode_avg_volatility) if marl_episode_avg_volatility else 0
    # 新增: 计算实验三指标
    average_inventory = total_inventory_sum / total_agent_steps if total_agent_steps > 0 else 0
    stockout_frequency = total_stockout_event_count / total_agent_steps if total_agent_steps > 0 else 0

    print("<<<<< MARL RESULTS START >>>>>") # 添加开始标记
    print(f"\n--- MAPPO-RNN 评估结果 ---")
    print(f"平均原始系统回报: {avg_marl_reward:.2f}")
    # 打印平均个体回报
    ind_rewards_str = ", ".join([f"Agent {i}: {r:.2f}" for i, r in enumerate(avg_marl_individual_rewards)])
    print(f"平均原始个体回报: [{ind_rewards_str}]")
    # 新增: 打印平均塑形奖励
    print(f"平均塑形系统回报: {avg_marl_shaped_reward:.2f}")
    ind_shaped_rewards_str = ", ".join([f"Agent {i}: {r:.2f}" for i, r in enumerate(avg_marl_individual_shaped_rewards)])
    print(f"平均塑形个体回报: [{ind_shaped_rewards_str}]")
    # 新增: 打印平均价格波动性
    print(f"平均价格波动性 (Std Dev): {avg_marl_volatility:.4f}")
    # 新增: 打印实验三指标
    print(f"平均库存水平: {average_inventory:.2f}")
    print(f"缺货频率: {stockout_frequency:.4f}")
    print("----------------------------")
    print("<<<<< MARL RESULTS END >>>>>") # 添加结束标记

    # --- 评估基线策略 ---
    print("\n--- 评估基线策略 ---")
    baseline_policies = ['fixed_low', 'fixed_high', 'random']
    baseline_results = {}

    for policy_type in baseline_policies:
        print(f"  开始评估基线: {policy_type}...")
        policy_total_rewards = []
        for ep in range(args.eval_episodes):
            # 注意: 这里使用同一个 env 实例，run_baseline_episode 内部会调用 env.reset()
            episode_reward = run_baseline_episode(
                env,
                policy_type=policy_type,
                episode_len=args.episode_len,
                n_agents=args.n_agents,
                n_price_levels=args.n_price_levels
            )
            policy_total_rewards.append(episode_reward)
            # 打印基线评估进度 (可选)
            # if (ep + 1) % max(1, args.eval_episodes // 10) == 0 or args.eval_episodes <= 10:
            #     print(f"    {policy_type.capitalize()} Episode {ep+1}/{args.eval_episodes} 完成, 原始奖励: {episode_reward:.2f}")

        avg_policy_reward = np.mean(policy_total_rewards) if policy_total_rewards else 0
        baseline_results[policy_type] = avg_policy_reward
        print(f"  完成评估基线: {policy_type}, 平均原始系统回报: {avg_policy_reward:.2f}")

    print("\n--- 基线策略评估结果汇总 ---")
    for policy, reward in baseline_results.items():
        print(f"{policy.replace('_', ' ').title()}: {reward:.2f}")
    print("------------------------------")

    # --- 可视化 (使用第一个 MARL 周期的详细数据) ---
    print("\n--- 生成 MARL 模型详细评估图 (基于第一个评估周期) ---")
    # 转换数据为 NumPy 数组
    try:
        # 确保 first_episode_data['prices'] 已经填充
        eval_prices_arr = np.array(first_episode_data['prices'])
        eval_obs_target_arr = np.array(first_episode_data['obs_target'])
        eval_sales_target_arr = np.array(first_episode_data['sales_target'])
        # 确保库存数据长度与其他数据匹配以进行绘图
        plot_len = eval_prices_arr.shape[0] if eval_prices_arr.size > 0 else 0 # Handle empty prices
        eval_inventory_target_arr = np.array(first_episode_data['inventory_target'][:plot_len]) if plot_len > 0 else np.array([])

        # 数据有效性检查
        if plot_len == 0 or not (
                eval_prices_arr.size > 0 and eval_prices_arr.shape[0] == plot_len
                and eval_obs_target_arr.size > 0 and eval_obs_target_arr.shape[0] == plot_len
                and eval_sales_target_arr.size > 0 and eval_sales_target_arr.shape[0] == plot_len
                and eval_inventory_target_arr.size == 0 or eval_inventory_target_arr.shape[0] == plot_len):
            raise ValueError("数据长度不匹配或为空")

        plot_all_agent_prices(eval_prices_arr,
                              args.n_agents,
                              args.n_price_levels,
                              save_dir)

        plot_agent_details_inv(eval_prices_arr, eval_obs_target_arr,
                              eval_sales_target_arr, eval_inventory_target_arr,
                              target_agent_id=args.target_agent_id,
                              n_price_levels=args.n_price_levels,
                              initial_inventory=args.initial_inventory,
                              save_dir=save_dir,
                              max_capacity=args.max_inventory_capacity,
                              episode_length=plot_len)

        # 调用新的绘图函数
        plot_price_distribution_heatmap(eval_prices_arr,
                                         args.n_agents,
                                         args.n_price_levels,
                                         save_dir)

        plot_price_coordination(eval_prices_arr, save_dir)

    except ValueError as ve:
        print(f"警告: 无法生成详细图表，数据存在问题: {ve}")
        print(f"  数据长度: Prices={len(first_episode_data['prices'])}, Obs={len(first_episode_data['obs_target'])}, Sales={len(first_episode_data['sales_target'])}, Inv={len(first_episode_data['inventory_target'])}")
    except Exception as e:
        print(f"警告: 生成图表时发生未知错误: {e}")

    print("--- 评估脚本执行完毕 ---")
