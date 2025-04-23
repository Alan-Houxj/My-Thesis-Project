import numpy as np
import heapq # 用于管理按到达时间排序的订单

class PricingEnvironment:
    """
    模拟多个电商产品的动态定价环境。
    """
    def __init__(self, n_products, n_price_levels=10, episode_length=200, homogeneous=False,
                 shock_prob=0.05, shock_duration=10, shock_magnitude=0.6,
                 initial_inventory=300, holding_cost=0.01,
                 replenishment_threshold=150, # 新增: 补货阈值
                 replenishment_amount=400,    # 使用调整后的补货量
                 max_inventory_capacity=800,    # 使用调整后的上限
                 replenishment_lead_time=10, # 新增: 补货提前期
                 stockout_penalty_per_unit=1.0,
                 omega=1.0): # 新增: omega 参数，默认为 1 (纯个体奖励)
        """
        初始化环境 (含库存和带提前期的阈值补货)。

        Args:
            n_products (int): 产品（智能体）的数量。
            n_price_levels (int): 每个产品可选择的离散价格等级数量。
            episode_length (int): 每个训练周期的最大时间步数。
            homogeneous (bool): 如果为 True，所有智能体使用相同参数。
            shock_prob (float): 需求冲击的概率。
            shock_duration (int): 需求冲击的持续时间。
            shock_magnitude (float): 需求冲击的影响幅度。
            initial_inventory (int): 初始库存水平。
            holding_cost (float): 单位持有成本。
            replenishment_threshold (int): 补货阈值。
            replenishment_amount (int): 每次补货量。
            max_inventory_capacity (int): 库存容量上限。
            replenishment_lead_time (int): 补货提前期。
            stockout_penalty_per_unit (float): 单位缺货惩罚。
            omega (float): 奖励塑造参数，默认为 1 (纯个体奖励)
        """
        self.n_products = n_products
        self.n_price_levels = n_price_levels
        self.episode_length = episode_length
        self.homogeneous = homogeneous # 存储标志
        self.shock_prob = shock_prob
        self.shock_duration = shock_duration
        self.shock_magnitude = shock_magnitude
        # 修改: 存储库存参数
        self.initial_inventory_level = initial_inventory # 用于重置和归一化
        self.holding_cost_per_unit = holding_cost
        # 新增: 存储阈值参数
        self.replenishment_threshold = replenishment_threshold
        self.replenishment_amount = replenishment_amount
        self.max_inventory_capacity = max_inventory_capacity # 设置一个上限防止无限累积
        # 新增: 存储提前期
        self.replenishment_lead_time = replenishment_lead_time
        # 新增: 存储缺货惩罚参数
        self.stockout_penalty_per_unit = stockout_penalty_per_unit
        # 新增: 存储 omega
        self.omega = omega
        if not (0.0 <= self.omega <= 1.0):
            raise ValueError(f"Omega must be between 0.0 and 1.0, got {self.omega}")

        # 修改: 增加观察维度以包含对手平均价格和库存信息
        self.action_space = [n_price_levels] * n_products
        self.observation_dim = 4 # [my_price_norm, time_norm, avg_competitor_price_norm, inventory_norm]
        # 修改: 状态维度也相应增加 (因为是观察的拼接)
        self.state_dim = self.observation_dim * self.n_products

        # --- 市场参数 (示例，可以从 config 加载) ---
        # 修改: 根据 homogeneous 标志生成参数
        if self.homogeneous:
            print("--- 使用同质智能体参数 ---")
            # 为第一个智能体生成参数
            cost_0 = np.random.uniform(0.1, 0.8)
            base_demand_0 = np.random.uniform(50, 100)
            price_elasticity_0 = np.random.uniform(-15, -8)
            cross_elasticity_row_0 = np.random.uniform(0.8, 1.5, n_products)
            cross_elasticity_row_0[0] = 0 # 对角线为 0

            # 复制给所有智能体
            self.costs = np.full(n_products, cost_0)
            self.base_demands = np.full(n_products, base_demand_0)
            self.price_elasticities = np.full(n_products, price_elasticity_0)
            # 交叉弹性矩阵需要对称（或至少对所有智能体结构相同）
            # 这里简单地让每一行都一样（除了对角线）
            self.cross_elasticities = np.tile(cross_elasticity_row_0, (n_products, 1))
            np.fill_diagonal(self.cross_elasticities, 0)

        else:
            print("--- 使用异质智能体参数 ---")
            # 使用原来的异质参数生成逻辑
            self.costs = np.random.uniform(0.1, 0.8, n_products)
            print(f"产品成本已初始化: {self.costs}") # 打印成本以便查看
            print(f"产品成本: {np.round(self.costs, 2)}")

            # 基础需求 (每个产品在价格最低时的需求)
            self.base_demands = np.random.uniform(50, 100, n_products)
            # 价格弹性 (价格每升高一级，需求减少的数量) - 必须是负数
            # 这里简化，弹性系数乘以价格等级 (1 to n_levels)
            self.price_elasticities = np.random.uniform(-15, -8, n_products)
            print(f"价格弹性已初始化: {self.price_elasticities}") # 打印弹性系数
            print(f"基础需求: {np.round(self.base_demands, 2)}")
            print(f"价格弹性: {np.round(self.price_elasticities, 2)}")

            # 修改: 激活并初始化交叉价格弹性
            # 假设产品间是替代品，交叉弹性为正
            # C[i, j] > 0 表示产品 j 价格升高，产品 i 需求增加
            self.cross_elasticities = np.random.uniform(0.8, 1.5, (n_products, n_products)) # 基础范围
            # 对角线设为 0 (自身价格影响由 price_elasticities 处理)
            np.fill_diagonal(self.cross_elasticities, 0)
            # 可以考虑归一化或调整强度，避免效应过大
            # self.cross_elasticities /= n_products # (可选) 平均化影响
            print(f"交叉价格弹性矩阵已初始化:\n{self.cross_elasticities}")
            print(f"交叉弹性矩阵:\n{np.round(self.cross_elasticities, 2)}")

        # --- 环境状态 ---
        self.current_step = 0
        # 存储每个智能体当前选择的价格等级 (从 0 到 n_price_levels-1)
        self.current_prices_idx = np.zeros(self.n_products, dtype=int)
        # 修改: 添加库存状态
        self.current_inventory = np.full(n_products, float(self.initial_inventory_level)) # 使用浮点数以防万一

        # 冲击状态变量
        self.shock_active = False
        self.shock_timer = 0
        # 冲击目标: -1 表示所有, 0, 1, ... 表示特定智能体
        self.shock_target_agent = -1

        # 新增: 跟踪在途补货订单
        # 使用最小堆存储 (到达时间, agent_id, 数量)，方便快速找到最早到达的
        self.pending_replenishments = []
        # 跟踪每个 agent 是否已有在途订单，避免重复下单
        self.order_pending_for_agent = [False] * n_products

        print(f"初始库存水平: {self.initial_inventory_level}")
        print(f"单位持有成本: {self.holding_cost_per_unit}")
        print(f"补货阈值: {self.replenishment_threshold}, 补货量: {self.replenishment_amount}, 提前期: {self.replenishment_lead_time}")
        print(f"最大库存容量: {self.max_inventory_capacity}")
        print(f"单位缺货惩罚: {self.stockout_penalty_per_unit}")
        print(f"个体奖励权重 Omega: {self.omega}") # 打印 Omega

    def _price_level_to_value(self, price_level_idx):
        """将价格等级索引 (0 to n-1) 转换为模拟价格值 (例如 1 to n)。"""
        # 简单映射：等级 0 -> 价格 1, 等级 n-1 -> 价格 n
        return price_level_idx + 1

    def _calculate_demand(self, price_values):
        """计算需求（含交叉弹性、噪声，并应用冲击）"""
        demands = np.zeros(self.n_products)
        for i in range(self.n_products):
            # 基础需求部分 (带噪声的正弦波动)
            time_factor = np.sin(np.pi * self.current_step / self.episode_length)
            noise_level = 0.3
            noise = np.random.normal(0, noise_level)
            noisy_time_factor = np.clip(time_factor + noise, -1, 1)
            amplitude = 0.5
            current_base_demand_i = self.base_demands[i] * (1 + amplitude * noisy_time_factor)

            # --- 应用需求冲击 ---
            if self.shock_active and (self.shock_target_agent == -1 or self.shock_target_agent == i):
                # 冲击效果是基础需求的一个比例
                shock_effect = self.base_demands[i] * self.shock_magnitude
                current_base_demand_i += shock_effect
                current_base_demand_i = max(0, current_base_demand_i) # 确保不为负
            # --- 冲击应用结束 ---

            # 自身和交叉价格影响 (不变)
            demand_i = current_base_demand_i + self.price_elasticities[i] * price_values[i]
            cross_effect = 0
            for j in range(self.n_products):
                if i != j:
                    cross_effect += self.cross_elasticities[i, j] * price_values[j]
            demand_i += cross_effect

            # 最终需求 (带少量基础噪声)
            demands[i] = max(0, demand_i + np.random.normal(0, 1))

        return demands

    def _get_obs(self):
        """
        为每个智能体生成局部观察，包含对手平均价格和库存信息。
        """
        observations = []
        max_price_level = self.n_price_levels - 1 # 最大价格等级索引
        if max_price_level <= 0: max_price_level = 1 # 避免除零
        # 使用初始库存作为最大值进行归一化
        max_inventory = self.initial_inventory_level
        if max_inventory <= 0: max_inventory = 1 # 避免除零

        all_prices = self.current_prices_idx # 当前所有智能体的价格等级

        for i in range(self.n_products):
            # 自己的价格 (归一化)
            my_price_norm = all_prices[i] / max_price_level

            # 时间步 (归一化)
            time_norm = self.current_step / self.episode_length

            # 对手平均价格 (归一化)
            competitor_prices = np.delete(all_prices, i) # 获取除自己外的所有价格
            if len(competitor_prices) > 0:
                avg_competitor_price = np.mean(competitor_prices)
                avg_competitor_price_norm = avg_competitor_price / max_price_level
            else: # 如果只有一个智能体
                avg_competitor_price_norm = 0

            # 修改: 添加归一化库存信息
            inventory_norm = self.current_inventory[i] / max_inventory
            inventory_norm = np.clip(inventory_norm, 0, 1) # 确保在 0-1 之间

            # 构建观察向量
            obs_i = np.array([my_price_norm, time_norm, avg_competitor_price_norm, inventory_norm])
            observations.append(obs_i)
        return observations

    def _get_state(self):
        """
        生成全局状态 (所有局部观察的拼接)。
        维度将自动变为 n_agents * obs_dim。
        """
        return np.concatenate(self._get_obs())

    def reset(self):
        """
        重置环境到初始状态。

        Returns:
            tuple: (initial_observations, initial_state)
                   - initial_observations (list): 每个智能体的初始局部观察列表。
                   - initial_state (np.ndarray): 初始全局状态。
        """
        self.current_step = 0
        # 初始价格可以随机或设为最低/中间值
        self.current_prices_idx = np.random.randint(0, self.n_price_levels, self.n_products)
        # self.current_prices_idx = np.zeros(self.n_products, dtype=int) # 从最低价开始

        # 重置冲击状态
        self.shock_active = False
        self.shock_timer = 0
        self.shock_target_agent = -1

        # 重置库存
        self.current_inventory.fill(float(self.initial_inventory_level))

        # 修改: 清空在途订单
        self.pending_replenishments = []
        self.order_pending_for_agent = [False] * self.n_products

        initial_observations = self._get_obs()
        initial_state = self._get_state()
        return initial_observations, initial_state

    def step(self, actions):
        """
        环境执行一个时间步。

        Args:
            actions (list or np.ndarray): 包含每个智能体选择的动作（价格等级索引 0 to n-1）的列表或数组。

        Returns:
            tuple: (next_observations, next_state, rewards, dones, info)
                   - next_observations (list): 每个智能体的下一个局部观察。
                   - next_state (np.ndarray): 下一个全局状态。
                   - rewards (list): 每个智能体获得的奖励列表。
                   - dones (list): 指示每个智能体是否结束的布尔值列表（在此环境中，所有智能体同时结束）。
                   - info (dict): 额外信息（例如实际需求量），可选。
        """
        actions = np.array(actions, dtype=int)
        if actions.shape != (self.n_products,):
             raise ValueError(f"动作数量必须为 {self.n_products}, 实际为: {actions.shape}")
        if np.any(actions < 0) or np.any(actions >= self.n_price_levels):
            raise ValueError(f"动作必须在 [0, {self.n_price_levels - 1}] 范围内")

        # --- 修改: 1. 检查并处理本步到达的补货 ---
        arrivals_processed = False
        inventory_before_arrival = self.current_inventory.copy()
        arrived_agent_ids = [] # 记录哪些 agent 在本步收到货
        # 使用 heapq 检查是否有订单在本步或之前到达
        while self.pending_replenishments and self.pending_replenishments[0][0] <= self.current_step:
            arrival_step, agent_id, amount = heapq.heappop(self.pending_replenishments)
            arrivals_processed = True
            arrived_agent_ids.append(agent_id)

            amount_to_add = amount
            # 检查容量限制
            if self.current_inventory[agent_id] + amount_to_add > self.max_inventory_capacity:
                amount_to_add = self.max_inventory_capacity - self.current_inventory[agent_id]
                amount_to_add = max(0, amount_to_add)

            if amount_to_add > 0:
                self.current_inventory[agent_id] += amount_to_add
                # 标记该 agent 的订单已完成，可以下新订单了
                self.order_pending_for_agent[agent_id] = False
                print(f"\n--- Step {self.current_step}: Replenishment ARRIVED for Agent {agent_id} ---")
                print(f"  Order placed at step: {arrival_step - self.replenishment_lead_time -1}") # 反推下单时间
                print(f"  Inventory Before Arrival: {inventory_before_arrival[agent_id]:.1f}")
                print(f"  Amount Added (capped at {self.max_inventory_capacity}): {amount_to_add:.1f}")
                print(f"  Inventory After Arrival: {self.current_inventory[agent_id]:.1f}\n")

        # --- 2. 冲击状态更新 ---
        if not self.shock_active:
            if np.random.rand() < self.shock_prob:
                self.shock_active = True
                self.shock_timer = self.shock_duration
                if self.n_products > 1 and np.random.rand() < 0.5:
                    self.shock_target_agent = np.random.randint(self.n_products)
                else:
                    self.shock_target_agent = -1
                # 注释掉冲击开始的打印
                # print(f"\n*** Demand Shock START - Target: {'All' if self.shock_target_agent == -1 else f'Agent {self.shock_target_agent}'}, Duration: {self.shock_timer}, Magnitude: {self.shock_magnitude:.2f} ***\n")
        elif self.shock_active:
            self.shock_timer -= 1
            if self.shock_timer <= 0:
                self.shock_active = False
                self.shock_target_agent = -1
                # 注释掉冲击结束的打印
                # print("\n*** Demand Shock END ***\n")
        # --- 冲击状态更新结束 ---

        self.current_prices_idx = actions
        # 将价格等级转换为实际价格值
        current_price_values = self._price_level_to_value(self.current_prices_idx)

        # 计算需求
        potential_demands = self._calculate_demand(current_price_values)

        # --- 4. 处理销售和库存更新 ---
        actual_sales = np.zeros_like(potential_demands)
        for i in range(self.n_products):
            # 销售量不能超过当前库存
            sales_i = min(potential_demands[i], self.current_inventory[i])
            sales_i = max(0, sales_i) # 确保销售不为负
            actual_sales[i] = sales_i
            # 更新库存
            self.current_inventory[i] -= sales_i

        # --- 5. 计算奖励 (利润 - 持有成本 - 缺货惩罚) ---
        profits = (current_price_values - self.costs) * actual_sales # 利润基于实际销售
        holding_penalties = self.holding_cost_per_unit * self.current_inventory # 计算持有成本

        # 新增: 计算缺货惩罚
        unsatisfied_demand = np.maximum(0, potential_demands - actual_sales) # 计算未满足需求
        stockout_penalties = self.stockout_penalty_per_unit * unsatisfied_demand

        # 计算原始个体奖励
        original_rewards = profits - holding_penalties - stockout_penalties

        # 计算系统总奖励 (原始奖励之和)
        system_reward = np.sum(original_rewards)

        # 应用 Omega 进行奖励塑造
        new_rewards = []
        for r_i in original_rewards:
            shaped_r = self.omega * r_i + (1.0 - self.omega) * system_reward
            new_rewards.append(shaped_r)
        rewards = new_rewards # 使用塑形后的奖励

        # --- 6. 修改: 在步骤结束时检查阈值并下新订单 ---
        inventory_after_sales = self.current_inventory.copy() # 记录销售后的库存
        order_placed_for = []
        for i in range(self.n_products):
            # 检查是否低于阈值 且 没有在途订单
            if self.current_inventory[i] < self.replenishment_threshold and not self.order_pending_for_agent[i]:
                order_placed_for.append(i)
                # 计算预计到达时间 (当前步+1 + 提前期)
                arrival_step = self.current_step + 1 + self.replenishment_lead_time
                amount = self.replenishment_amount
                # 加入在途订单列表 (使用 heapq)
                heapq.heappush(self.pending_replenishments, (arrival_step, i, amount))
                # 标记该 agent 已有在途订单
                self.order_pending_for_agent[i] = True
                print(f"\n--- Step {self.current_step}: Replenishment ORDER PLACED for Agent {i} ---")
                print(f"  Inventory After Sales: {inventory_after_sales[i]:.1f} (< Threshold {self.replenishment_threshold})")
                print(f"  Order Amount: {amount}, Expected Arrival Step: {arrival_step}\n")

        # --- 7. 更新时间和状态 ---
        self.current_step += 1
        next_observations = self._get_obs()
        next_state = self._get_state()

        # 检查是否结束
        is_done = self.current_step >= self.episode_length
        dones = [is_done] * self.n_products # 所有智能体同时结束

        # 修改: 更新 info 字典，加入缺货信息
        info = {
            'potential_demands': potential_demands,
            'actual_sales': actual_sales,
            'unsatisfied_demand': unsatisfied_demand, # 新增
            'stockout_penalties': stockout_penalties, # 新增
            'prices': current_price_values,
            'costs': self.costs,
            'inventory': self.current_inventory.copy(), # 返回的是本步所有操作后的期末库存
            'holding_penalties': holding_penalties,
            'original_rewards': original_rewards.tolist(), # 加入原始奖励
            'shaped_rewards': rewards # 加入塑形后奖励 (虽然它也是主返回值)
        }

        return next_observations, next_state, rewards, dones, info

    def get_env_info(self):
        """返回环境的基本信息，用于智能体初始化。"""
        return {
            "n_agents": self.n_products,
            "obs_shape": self.observation_dim,
            "state_shape": self.state_dim,
            "n_actions": self.n_price_levels,
            "episode_limit": self.episode_length
        }

# --- 示例用法 ---
if __name__ == '__main__':
    n_agents = 3
    n_levels = 5
    env = PricingEnvironment(n_products=n_agents, n_price_levels=n_levels, episode_length=120,
                             initial_inventory=150, holding_cost=0.01,
                             replenishment_threshold=50, replenishment_amount=100, max_inventory_capacity=200,
                             stockout_penalty_per_unit=1.0) # 测试阈值和缺货惩罚
    env_info = env.get_env_info()
    print("环境信息 (含库存):", env_info)

    obs, state = env.reset()
    print(f"初始库存: {env.current_inventory}")
    print("-" * 20)

    done = [False] * n_agents
    total_reward = np.zeros(n_agents)
    step_count = 0

    while not all(done):
        step_count += 1
        # 随机选择动作 (模拟智能体决策)
        actions = np.random.randint(0, n_levels, n_agents)
        print(f"步骤 {step_count}, 动作 (价格等级): {actions}")

        next_obs, next_state, rewards, dones, info = env.step(actions)

        total_reward += rewards
        print(f"  价格: {info['prices']}")
        print(f"  潜在需求: {np.round(info['potential_demands'], 2)}")
        print(f"  实际销售: {np.round(info['actual_sales'], 2)}")
        print(f"  期末库存: {np.round(info['inventory'], 2)}")
        print(f"  持有惩罚: {np.round(info['holding_penalties'], 2)}")
        print(f"  未满足需求: {np.round(info['unsatisfied_demand'], 2)}")
        print(f"  缺货惩罚: {np.round(info['stockout_penalties'], 2)}")
        print(f"  奖励(利润-持有-缺货): {np.round(rewards, 2)}")
        print(f"  结束: {dones}")
        print("-" * 10)

        obs = next_obs
        state = next_state

        if step_count > env.episode_length + 5: # 安全退出
             print("超过最大步数，强制退出")
             break

    print(f"\n周期结束，总步数: {env.current_step}")
    print(f"最终库存: {np.round(env.current_inventory, 2)}")
    print(f"总奖励: {np.round(total_reward, 2)}")
