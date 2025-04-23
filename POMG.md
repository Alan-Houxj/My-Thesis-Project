# 项目 POMG (Partially Observable Markov Game) 详细描述

## 1. 引言

本项目模拟了一个多智能体动态定价与库存管理的市场环境，可以被精确地建模为一个**部分可观察马尔可夫博弈 (POMG)**。此模型包含多个以利润最大化（或经塑造后的奖励最大化）为目标的智能体（产品），它们在共享环境中交互，但每个智能体只能获取部分环境信息。

**核心文件参考:**

*   `environment.py`: 定义了 POMG 的状态空间、动作空间、转移函数、奖励函数和观测函数的核心逻辑。
*   `agent.py`: 定义了智能体的策略表示（Actor 网络）和价值函数估计（Critic 网络），以及学习算法（MAPPO）。
*   `main.py`: 实现了 POMG 的求解过程（通过 MAPPO 训练智能体策略）。
*   `evaluate.py`: 用于加载和评估训练得到的策略在 POMG 中的表现。

---

## 2. POMG 组件定义

一个 POMG 通常由以下元组定义： $(\mathcal{N}, \mathcal{S}, \mathcal{A}, T, \mathcal{R}, \Omega, O, \gamma, H)$

### 2.1. 智能体集合 (Agents) $\mathcal{N}$

*   **定义:** 博弈中的参与者集合。
*   **本项目:** 集合 $\mathcal{N} = \{1, 2, ..., N\}$，其中 $N$ 是产品的数量（智能体的数量）。
*   **代码实现:** 由 `environment.py` 中的 `self.n_products` 或 `main.py`/`evaluate.py` 中的 `args.n_agents` 定义。

### 2.2. 状态空间 (State Space) $\mathcal{S}$

*   **定义:** 描述环境完整、真实状态的集合。状态 $s \in \mathcal{S}$ 包含了做出下一个状态转移和奖励计算所需的所有信息，即使这些信息不一定对所有智能体可见。
*   **本项目:** 一个状态 $s_t \in \mathcal{S}$ 在时间步 $t$ 可以由以下组件构成：
    *   **当前时间步 (Current Step):** $t \in \{0, 1, ..., H\}$ (`self.current_step`)。
    *   **所有智能体的当前价格等级索引 (Current Price Levels):** $\mathbf{p}_{idx, t-1} = (p_{idx, 1, t-1}, ..., p_{idx, N, t-1})$。注意这里是 *上一步* 设定的价格，它会影响 *当前步* 的需求。(`self.current_prices_idx`，在`step`开始时代表上一步动作的结果)。
    *   **所有智能体的真实库存水平 (Current Inventory):** $\mathbf{I}_t = (I_{1,t}, ..., I_{N,t})$，其中 $I_{i,t} \in [0, I_{max}]$。(`self.current_inventory`)。
    *   **需求冲击状态 (Shock State):** 一个元组 $(\text{active}_t, \text{timer}_t, \text{target}_t)$，表示冲击是否激活、剩余持续时间、冲击目标智能体。(`self.shock_active`, `self.shock_timer`, `self.shock_target_agent`)。
    *   **在途补货订单 (Pending Replenishments):** 一个列表或优先队列，包含所有已下单但尚未到达的补货信息，例如 $[(\text{arrival\_step}_1, \text{agent\_id}_1, \text{amount}_1), ...]$。(`self.pending_replenishments`)。
    *   **智能体是否有在途订单标记 (Order Pending Flags):** 一个布尔向量 $(\text{pending}_{1,t}, ..., \text{pending}_{N,t})$，标记每个智能体当前是否有订单在途。(`self.order_pending_for_agent`)。
    *   **隐式状态:** 环境的固定参数，如成本 `self.costs`、基础需求 `self.base_demands`、价格弹性 `self.price_elasticities`、交叉弹性 `self.cross_elasticities`、补货参数 `self.replenishment_threshold`, `self.replenishment_amount`, `self.replenishment_lead_time` 等，虽然在单个 episode 内不变，但它们定义了具体的 POMG 实例及其动态。
*   **状态空间特性:** 状态空间是混合的（包含离散和连续变量），且维度较高。

### 2.3. 动作空间 (Action Space) $\mathcal{A}$

*   **定义:** 所有智能体可能采取的联合动作的集合 $\mathcal{A} = \times_{i \in \mathcal{N}} \mathcal{A}_i$，其中 $\mathcal{A}_i$ 是智能体 $i$ 的个体动作空间。
*   **本项目:**
    *   **个体动作空间 $\mathcal{A}_i$**: 每个智能体 $i$ 选择一个离散的价格等级索引。$\mathcal{A}_i = \{0, 1, ..., K-1\}$，其中 $K$ 是价格等级数量 (`self.n_price_levels`)。
    *   **联合动作空间 $\mathcal{A}$**: 一个联合动作 $\mathbf{a}_t = (a_{1,t}, ..., a_{N,t}) \in \mathcal{A}$ 是一个包含所有智能体在时间步 $t$ 选择的价格等级索引的向量。
*   **代码实现:** `env.step(actions)` 中的 `actions` 参数代表一个联合动作 $\mathbf{a}_t$。`env_info['n_actions']` 对应 $K$。

### 2.4. 转移函数 (Transition Function) $T$

*   **定义:** 定义了状态转换的概率。$T(s' | s, \mathbf{a}): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0, 1]$，表示在状态 $s$ 下执行联合动作 $\mathbf{a}$ 后，转移到状态 $s'$ 的概率。
*   **本项目:** 转移函数是**随机的 (Stochastic)**，由 `environment.py` 中的 `env.step()` 方法隐式定义，其过程复杂：
    1.  **补货到达:** 检查 `pending_replenishments`，更新对应智能体的 `current_inventory` (增加量受 `max_inventory_capacity` 限制) 和 `order_pending_for_agent`。
    2.  **冲击状态更新:** 根据 `shock_prob` 概率性触发新冲击或更新现有冲击的 `shock_timer`。这是随机性的来源之一。
    3.  **价格更新:** `current_prices_idx` 被设置为当前联合动作 $\mathbf{a}_t$。
    4.  **需求计算 (`_calculate_demand`):**
        *   计算受当前价格 `_price_level_to_value(actions)`、弹性、波动（正弦+噪声）、以及激活的冲击影响的潜在需求 $\mathbf{D}_t$。需求计算中的**随机噪声** (`np.random.normal`) 是随机性的主要来源之二。
    5.  **销售与库存更新:**
        *   计算实际销售量 $Sales_{i,t} = \min(D_{i,t}, I_{i,t})$。
        *   更新库存 $I_{i, t+1}' = I_{i,t} - Sales_{i,t}$ (这是中间步骤，还需加上述补货)。
    6.  **新补货下单:** 检查更新后的库存 $I_{i, t+1}'$ 是否低于 `replenishment_threshold` 且 `order_pending_for_agent[i]` 为 False。若是，则向 `pending_replenishments` 添加新订单（到达时间为 $t + 1 + L$），并设置 `order_pending_for_agent[i]` 为 True。
    7.  **时间步更新:** $t \leftarrow t + 1$。
    *   **总结:** 转移函数 $T$ 是高度非线性的，且由于需求噪声和冲击触发而具有随机性。它依赖于完整的状态 $s$ 和联合动作 $\mathbf{a}$。

### 2.5. 奖励函数 (Reward Function) $\mathcal{R}$

*   **定义:** 为每个智能体分配一个实值奖励。$\mathcal{R}(s, \mathbf{a}): \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}^N$，给出在状态 $s$ 执行联合动作 $\mathbf{a}$ 后，每个智能体 $i$ 获得的奖励 $r_{i}$。 (有时也定义为 $\mathcal{R}(s, \mathbf{a}, s')$)。
*   **本项目:** 奖励函数也在 `env.step()` 中计算，并且使用了**奖励塑造 (Reward Shaping)**：
    1.  **计算利润 $\pi_{i,t}$**:
        $ \pi_{i,t} = (p_{i,t} - \text{cost}_i) \times Sales_{i,t} $
        其中 $p_{i,t}$ 是由动作 $a_{i,t}$ 转换的价格值。
    2.  **计算持有成本惩罚 $P_{hold, i, t}$**:
        $ P_{hold, i, t} = c_h \times I_{i, t+1} $
        其中 $I_{i, t+1}$ 是 *完成所有更新后* 的期末库存。
    3.  **计算缺货惩罚 $P_{stockout, i, t}$**:
        $ P_{stockout, i, t} = c_s \times \max(0, D_{i,t} - Sales_{i,t}) $
    4.  **计算原始个体奖励 $r_{i,t}$**:
        $$ r_{i,t} = \pi_{i,t} - P_{hold, i, t} - P_{stockout, i, t} $$
    5.  **计算系统总奖励 $R_{sys, t}$**:
        $$ R_{sys, t} = \sum_{j=1}^N r_{j,t} $$
    6.  **计算塑形后奖励 $r'_{i,t}$ (用于学习)**:
        $$ r'_{i,t} = \omega \cdot r_{i,t} + (1 - \omega) \cdot R_{sys, t} $$
        其中 $\omega$ 是 `self.omega` 参数 (`args.omega`)。
    *   **最终奖励向量:** $\mathbf{r}'_t = (r'_{1,t}, ..., r'_{N,t})$。这个奖励依赖于状态 $s$，联合动作 $\mathbf{a}$，以及转移过程中的随机结果（需求）。

### 2.6. 观测空间 (Observation Space) $\Omega$

*   **定义:** 所有智能体可能接收到的联合观测的集合 $\Omega = \times_{i \in \mathcal{N}} \Omega_i$，其中 $\Omega_i$ 是智能体 $i$ 的个体观测空间。
*   **本项目:**
    *   **个体观测空间 $\Omega_i$**: 每个智能体 $i$ 接收一个包含部分信息的向量 $o_{i,t} \in \mathbb{R}^4$。
        $ o_{i,t} = [\text{norm}(p_{i,t}), \text{norm}(t), \text{norm}(\bar{p}_{-i, t}), \text{norm}(I_{i,t})] $
        其中：
        *   $\text{norm}(p_{i,t})$: 智能体自身价格的归一化值。
        *   $\text{norm}(t)$: 当前时间步的归一化值。
        *   $\text{norm}(\bar{p}_{-i, t})$: **竞争对手**平均价格的归一化值 ($\bar{p}_{-i, t} = \frac{1}{N-1}\sum_{j \neq i} p_{j,t}$)。
        *   $\text{norm}(I_{i,t})$: 智能体自身库存水平的归一化值。
    *   **归一化:** 使用 `self.n_price_levels - 1`, `self.episode_length`, `self.initial_inventory_level` 作为各自的最大值进行归一化，并裁剪到 [0, 1]。
    *   **部分可观察性:** 这是 POMG 的关键。智能体**不能**直接观察到：
        *   完整的状态 $s_t$。
        *   竞争对手的确切价格 $p_{j,t}$ (只能看到平均值)。
        *   竞争对手的库存 $I_{j,t}$。
        *   需求冲击的具体状态。
        *   在途订单的详细信息。
        *   环境的精确参数（弹性、成本等，除非是 `homogeneous` 情况且智能体知道）。

### 2.7. 观测函数 (Observation Function) $O$

*   **定义:** 定义了在转移到新状态 $s'$ 后接收到联合观测 $\mathbf{o}$ 的概率。$O(\mathbf{o} | s', \mathbf{a}): \Omega \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$。通常简化为基于结果状态 $s'$。
*   **本项目:** 观测函数由 `environment.py` 中的 `_get_obs()` 方法实现，并在 `env.step()` 的末尾调用。给定下一个状态 $s_{t+1}$（特别是其中的 `current_prices_idx` 和 `current_inventory` 以及 `current_step`），观测函数是**确定性的 (Deterministic)**，它直接根据 $s_{t+1}$ 计算出每个智能体的观测向量 $o_{i, t+1}$。联合观测 $\mathbf{o}_{t+1} = (o_{1, t+1}, ..., o_{N, t+1})$。

### 2.8. 折扣因子 (Discount Factor) $\gamma$

*   **定义:** 用于对未来奖励进行折现的因子 $\gamma \in [0, 1)$。
*   **本项目:** 在 `agent.py` 中用于计算 GAE 和回报目标。
*   **代码实现:** `args.gamma` (在 `main.py` 解析)，`self.gamma` (在 `MAPPOAgent` 中)。

### 2.9. 博弈时域 (Horizon) $H$

*   **定义:** 博弈的最大持续时间步数。
*   **本项目:** 博弈是**有限时域 (Finite Horizon)** 的，最大长度为 $H$。
*   **代码实现:** `args.episode_len` (在 `main.py`/`evaluate.py` 解析)，`self.episode_length` (在 `PricingEnvironment` 中)。

---

## 3. 智能体策略与目标

*   **策略 (Policy) $\pi_i$**: 每个智能体 $i$ 的目标是学习一个策略 $\pi_i(a_i | h_i)$，该策略根据其**历史观测** $h_i = (o_{i,0}, a_{i,0}, ..., o_{i,t})$ （在 RNN 实现中，历史信息被压缩到 LSTM 的隐藏状态 $(h_{lstm}, c_{lstm})$ 中）来输出选择下一个动作 $a_i$ 的概率分布。
    *   **代码实现:** 智能体的策略 $\pi_i$ 由 `agent.py` 中的 `Actor` 网络表示。`choose_action` 方法实现了根据当前观测 $o_{i,t}$ 和 LSTM 隐藏状态选择动作的过程。
*   **目标 (Objective):** 找到一个联合策略 $(\pi_1, ..., \pi_N)$，使得每个智能体 $i$ 的期望累积折扣（塑形后）奖励最大化：
    $$ J_i(\pi_1, ..., \pi_N) = \mathbb{E}_{\mathbf{a} \sim \pi, s' \sim T, \mathbf{o} \sim O} \left[ \sum_{t=0}^{H-1} \gamma^t r'_{i,t} | b_0 \right] $$
    其中 $b_0$ 是初始状态分布（由 `env.reset()` 定义）。MAPPO 算法旨在找到一个近似的纳什均衡或其他形式的稳定联合策略。
*   **价值函数 (Value Function):** 中心化的 Critic 网络 (`agent.py` 中的 `Critic`) 学习全局状态价值函数 $V_\phi(s_t)$，用于辅助 Actor 的训练（通过计算优势 $A_t$）。

---

## 4. 总结

该项目通过 `environment.py` 精确定义了一个复杂的 POMG，具有随机状态转移（需求噪声、冲击）和部分可观察性（受限的观测向量）。智能体（`agent.py`）使用带 RNN 的 Actor-Critic 架构 (MAPPO) 来处理部分可观察性和时序依赖。训练过程（`main.py`）的目标是学习能够在这个 POMG 中获得高累积奖励的分布式策略。`evaluate.py` 则用于检验学习到的策略在模拟博弈中的实际表现。
