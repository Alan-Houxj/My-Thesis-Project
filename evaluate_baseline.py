import torch
import numpy as np
import os
import argparse

from environment import PricingEnvironment
# Removed agent import as we don't load models

def run_fixed_strategy_episode(env, episode_len, n_agents, n_price_levels, fixed_actions):
    """Runs one episode with fixed actions for all agents.

    Args:
        env: The PricingEnvironment instance.
        episode_len (int): Max steps for the episode.
        n_agents (int): Number of agents.
        n_price_levels (int): Number of price levels (not directly used here but kept for context).
        fixed_actions (list): A list of fixed actions, one for each agent.

    Returns:
        list: List of total original individual rewards for each agent this episode.
    """
    obs_list, state = env.reset()
    done = False
    episode_original_rewards_per_agent = [0.0] * n_agents
    step = 0

    while not done and step < episode_len:
        # Use the provided fixed actions
        actions = fixed_actions

        next_obs_list, next_state, shaped_rewards, dones, info = env.step(actions)

        # IMPORTANT: Accumulate ORIGINAL individual rewards
        original_rewards = info.get('original_rewards', [0.0] * n_agents) # Default if key missing
        for i in range(n_agents):
            episode_original_rewards_per_agent[i] += original_rewards[i]

        obs_list = next_obs_list
        state = next_state
        done = all(dones)
        step += 1

    return episode_original_rewards_per_agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Single Agent Baseline (Fixed Others)")

    # --- Environment parameters (copied from evaluate.py, should match training) ---
    # Reduced default eval episodes for faster baseline runs
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of episodes for averaging results")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--n_price_levels", type=int, default=5, help="Number of price levels")
    parser.add_argument("--episode_len", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--homogeneous", type=bool, default=False, help="Whether agents are homogeneous")
    parser.add_argument("--shock_prob", type=float, default=0.05, help="Shock probability")
    parser.add_argument("--shock_duration", type=int, default=10, help="Shock duration")
    parser.add_argument("--shock_magnitude", type=float, default=0.5, help="Shock magnitude")
    # Use the parameters from the last successful run (omega=1, modHC)
    parser.add_argument("--initial_inventory", type=int, default=500, help="Initial inventory")
    parser.add_argument("--holding_cost", type=float, default=0.1, help="Holding cost (from last run)") # Adjusted HC
    parser.add_argument("--replenishment_threshold", type=int, default=200, help="Replenishment threshold")
    parser.add_argument("--replenishment_amount", type=int, default=100, help="Replenishment amount")
    parser.add_argument("--max_inventory_capacity", type=int, default=800, help="Max inventory capacity")
    parser.add_argument("--replenishment_lead_time", type=int, default=5, help="Replenishment lead time")
    parser.add_argument("--stockout_penalty", type=float, default=0.01, help="Stockout penalty")
    parser.add_argument("--omega", type=float, default=1.0, help="Omega value (should be 1.0 for this test)") # Set omega=1.0

    # --- Baseline specific parameters ---
    parser.add_argument("--test_agent_id", type=int, default=0, help="ID of the agent being tested")
    parser.add_argument("--fixed_test_action", type=int, required=True, help="Fixed action for the test agent")
    parser.add_argument("--fixed_other_actions", nargs='+', type=int, default=[1, 3], help="Fixed actions for other agents (Agent 1, Agent 2)")
    parser.add_argument("--save_dir", type=str, default="./results_exp1_baseline_omega1_modHC", help="Directory for context (not saving plots here)") # Reference original run dir
    parser.add_argument("--device", type=str, default="cpu", help="Device (CPU is fine for evaluation)") # Force CPU

    args = parser.parse_args()

    # --- Validate parameters ---
    if args.test_agent_id < 0 or args.test_agent_id >= args.n_agents:
        raise ValueError(f"Invalid test_agent_id: {args.test_agent_id}")
    if len(args.fixed_other_actions) != args.n_agents - 1:
        raise ValueError(f"Expected {args.n_agents - 1} fixed actions for other agents, got {len(args.fixed_other_actions)}")
    if not (0 <= args.fixed_test_action < args.n_price_levels):
         raise ValueError(f"Fixed test action {args.fixed_test_action} out of bounds [0, {args.n_price_levels-1}]")
    for action in args.fixed_other_actions:
        if not (0 <= action < args.n_price_levels):
            raise ValueError(f"Fixed other action {action} out of bounds [0, {args.n_price_levels-1}]")

    # --- Settings ---
    device = torch.device(args.device) # Though not used for tensors here
    save_dir = args.save_dir # Mostly for reference
    if not os.path.exists(save_dir):
        print(f"Warning: Directory {save_dir} not found, creating it for consistency.")
        os.makedirs(save_dir)

    print(f"--- 开始单智能体基线评估 ---")
    print(f"测试智能体 ID: {args.test_agent_id}")
    print(f"测试智能体固定动作: {args.fixed_test_action}")
    print(f"其他智能体固定动作: {args.fixed_other_actions}")
    print(f"环境参数 (omega={args.omega}, HC={args.holding_cost})") # Show key params
    print(f"运行 {args.eval_episodes} 个评估周期进行平均。")

    # --- Initialize Environment ---
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
        omega=args.omega # Pass omega to env
    )
    env_info = env.get_env_info()

    # --- Construct the fixed action list for the episode ---
    all_fixed_actions = [0] * args.n_agents
    other_action_idx = 0
    for i in range(args.n_agents):
        if i == args.test_agent_id:
            all_fixed_actions[i] = args.fixed_test_action
        else:
            all_fixed_actions[i] = args.fixed_other_actions[other_action_idx]
            other_action_idx += 1

    print(f"本轮评估将使用的固定动作序列: {all_fixed_actions}")

    # --- Run Evaluation Episodes ---
    all_episode_rewards_per_agent = [] # List to store reward lists from each episode
    for ep in range(args.eval_episodes):
        episode_rewards = run_fixed_strategy_episode(
            env, args.episode_len, args.n_agents, args.n_price_levels, all_fixed_actions
        )
        all_episode_rewards_per_agent.append(episode_rewards)

        # Print progress
        if (ep + 1) % max(1, args.eval_episodes // 10) == 0 or args.eval_episodes <= 10:
             # Show reward for the test agent in this episode
             print(f"  Episode {ep+1}/{args.eval_episodes} 完成. Agent {args.test_agent_id} 奖励: {episode_rewards[args.test_agent_id]:.2f}")

    # --- Calculate and Print Average Rewards ---
    if not all_episode_rewards_per_agent:
        print("错误：没有收集到评估周期的奖励数据。")
    else:
        avg_rewards_per_agent = np.mean(all_episode_rewards_per_agent, axis=0)
        print("\n--- 平均个体奖励 ---")
        for agent_id in range(args.n_agents):
            print(f"  Agent {agent_id}: {avg_rewards_per_agent[agent_id]:.2f}")

        print(f"\n--- 关键结果 ---")
        print(f"测试智能体 (Agent {args.test_agent_id}) 在固定动作为 {args.fixed_test_action} 时,")
        print(f"同时其他智能体固定动作为 {args.fixed_other_actions} 时,")
        print(f"获得的平均个体奖励为: {avg_rewards_per_agent[args.test_agent_id]:.2f}")
        print("--------------------")

    print("\n--- 单智能体基线评估执行完毕 ---") 