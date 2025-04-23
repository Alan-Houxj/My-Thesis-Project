# 多智能体定价系统（含库存管理、补货提前期、缺货惩罚和奖励塑造）的数学形式化

## 1. 引言

本项目构建了一个多智能体强化学习 (MARL) 系统，用于模拟和优化多个智能体（例如，零售商）在动态环境中的定价策略。该环境具有库存管理、补货延迟、缺货惩罚，并引入了奖励塑造机制 (`omega`) 来影响智能体间的合作与竞争行为。智能体使用基于循环神经网络 (RNN) 的多智能体近端策略优化 (MAPPO) 算法进行学习。

## 2. 环境模型 (`PricingEnvironment`)

### 2.1 基本设定

*   **智能体集合 (Agents)**: \( \mathcal{N} = \{0, 1, ..., N-1\} \)，共 \(N\) 个智能体。
*   **时间步 (Time Steps)**: 离散时间步 \( t = 0, 1, 2, ..., T_{max} \)，其中 \(T_{max}\) 是每个 episode 的最大长度 (`episode_len`)。
*   **同质性 (Homogeneity)**: 系统可以设定为智能体同质 (`homogeneous=True`) 或异质 (`homogeneous=False`)。在当前评估中，通常关注异质情况下的表现，即使训练时可能同质。

### 2.2 状态空间 (State Space) / 观察空间 (Observation Space)

每个智能体 \(i \in \mathcal{N}\) 在时间步 \(t\) 接收一个局部观察 \(o_{i,t}\)。该观察通常包含以下（可能经过归一化的）信息：

*   **自身价格 (Own Price)**: \( p_{i, t-1} \) (上一步采取的价格，或一个编码)。
*   **平均竞争者价格 (Average Competitor Price)**: \( \bar{p}_{\neg i, t-1} = \frac{1}{N-1} \sum_{j \neq i} p_{j, t-1} \) (如果 \(N>1\))。
*   **自身库存水平 (Own Inventory)**: \( I_{i,t} \) (当前时间步开始时的库存)。
*   **时间步编码 (Time Step)**: \( t / T_{max} \) (表示当前在 episode 中的位置)。
*   **（可选）需求冲击指示器 (Shock Indicator)**: 一个表示当前是否处于需求冲击期间的标志。

我们将智能体 \(i\) 在时间步 \(t\) 的观察向量表示为 \( o_{i,t} \)。全局状态 \(s_t\) 包含所有智能体的局部信息和环境的真实状态（如所有库存、所有价格、冲击状态等），但智能体通常只能访问其局部观察 \(o_{i,t}\)。

### 2.3 行动空间 (Action Space)

每个智能体 \(i\) 在时间步 \(t\) 选择一个行动 \(a_{i,t}\)，该行动对应一个离散的价格水平。

*   **价格水平 (Price Levels)**: 设共有 \(K\) 个价格水平 (`n_price_levels`)。行动空间为 \( \mathcal{A}_i = \{0, 1, ..., K-1\} \)。
*   **实际价格 (Actual Price)**: 存在一个映射函数 \( \text{PriceMap}(a_{i,t}) = p_{i,t} \)，将行动（价格水平索引）映射到实际价格值。例如，Level 0 为最低价，Level \(K-1\) 为最高价。

联合行动表示为 \( \mathbf{a}_t = (a_{0,t}, a_{1,t}, ..., a_{N-1,t}) \)。
联合价格表示为 \( \mathbf{p}_t = (p_{0,t}, p_{1,t}, ..., p_{N-1,t}) \)。

### 2.4 环境动态 (Transition Dynamics)

环境根据当前状态和所有智能体的联合行动 \( \mathbf{a}_t \) (或联合价格 \( \mathbf{p}_t \)) 转换到下一个状态。

1.  **需求函数 (Demand Function)**: 每个智能体 \(i\) 面临的需求 \(D_{i,t}\) 取决于所有智能体的价格和可能的需求冲击。
    \[
    D_{i,t} = f_D(p_{0,t}, p_{1,t}, ..., p_{N-1,t}, \text{shock}_t)
    \]
    *   如果智能体间**无竞争替代关系**，需求函数可能简化为仅依赖自身价格：\( D_{i,t} = f_D(p_{i,t}, \text{shock}_t) \)。
    *   `shock_t` 是一个随机变量，以概率 `shock_prob` 触发，持续 `shock_duration`，影响需求函数 \(f_D\)。

2.  **实际销售 (Actual Sales)**: 智能体 \(i\) 的实际销售量 \(S_{i,t}\) 受当前库存限制。
    \[
    S_{i,t} = \min(I_{i,t}, D_{i,t})
    \]

3.  **库存更新 (Inventory Update)**: 库存根据销售量和到货量进行更新。
    \[
    I_{i, t+1} = I_{i,t} - S_{i,t} + \text{Arrivals}_{i,t+1}
    \]
    *   \( \text{Arrivals}_{i,t+1} \) 是在时间步 \(t+1\) 开始时到达的补货量。

4.  **补货机制 (Replenishment Mechanism)**:
    *   **补货阈值 (Threshold)**: \( R_{th} \) (`replenishment_threshold`)
    *   **补货量 (Amount)**: \( R_{amt} \) (`replenishment_amount`)
    *   **补货提前期 (Lead Time)**: \( L \) (`replenishment_lead_time`)
    *   如果在时间步 \(t\)，智能体 \(i\) 的期末预计库存 \( I_{i,t} - S_{i,t} \) 低于 \( R_{th} \)，并且当前没有在途订单 (outstanding order)，则下一个补货订单被触发。
    *   该订单包含 \( R_{amt} \) 单位的货物，并将在时间步 \( t + L \) 到达 (即计入 \( \text{Arrivals}_{i, t+L} \))。需要跟踪在途订单以避免重复订购。

### 2.5 奖励函数 (Reward Function)

1.  **个体利润 (Individual Profit)**:
    \[
    \text{Profit}_{i,t} = S_{i,t} \times (p_{i,t} - C)
    \]
    其中 \(C\) 是单位成本（假设为常数，未在参数中明确，但隐含存在）。 *注意：如果成本 \(C=0\)，则利润就是收入 \(S_{i,t} \times p_{i,t}\)。*

2.  **缺货惩罚 (Stockout Penalty)**:
    \[
    \text{Penalty}_{i,t} = P_{so} \times \max(0, D_{i,t} - I_{i,t})
    \]
    其中 \( P_{so} \) 是单位缺货惩罚 (`stockout_penalty_per_unit` 或 `stockout_penalty`)。

3.  **个体原始奖励 (Individual Raw Reward)**:
    \[
    r_{i,t} = \text{Profit}_{i,t} - \text{Penalty}_{i,t}
    \]

4.  **系统总奖励 (System Total Reward)**:
    \[
    R_{total, t} = \sum_{j=0}^{N-1} r_{j,t}
    \]

5.  **奖励塑造 (Reward Shaping)**: 使用参数 `omega` (\(\omega\)) 来调整个体奖励。
    \[
    r'_{i,t} = \omega \cdot r_{i,t} + (1 - \omega) \cdot R_{total, t}
    \]
    *   当 \(\omega = 1\)，智能体只关心自身原始奖励 (纯竞争/自利)。
    *   当 \(\omega < 1\)，智能体的奖励包含了部分系统总奖励，激励合作行为。
    *   当 \(\omega = 0\)，智能体只关心系统总奖励 (纯合作)。

智能体 \(i\) 在 MARL 算法中优化的目标是最大化其**塑造后奖励** \(r'_{i,t}\) 的期望累积和（通常是折扣累积和）。

## 3. 智能体模型 (MAPPO with RNN)

智能体使用 MAPPO 算法学习策略。MAPPO 是 PPO 算法在多智能体环境中的一种应用，通常结合**集中式训练、分散式执行 (CTDE)** 的思想，但在这里似乎主要基于局部观察。每个智能体 \(i\) 维护一个策略网络（Actor）和一个价值网络（Critic），两者都使用 RNN 来处理时序信息。

### 3.1 循环策略网络 (Recurrent Actor Network)

*   策略 \(\pi_{\theta_i}\) 由参数为 \(\theta_i\) 的神经网络（包含 RNN 层）表示。
*   输入：当前观察 \(o_{i,t}\) 和 RNN 的上一个隐藏状态 \(h_{i,t}\)。
*   输出：下一个隐藏状态 \(h_{i, t+1}\) 和在行动空间 \(\mathcal{A}_i\) 上的概率分布 \(\pi_{\theta_i}(\cdot | o_{i,t}, h_{i,t})\)。
    \[
    h_{i,t+1}, \pi_{\theta_i}(\cdot | o_{i,t}, h_{i,t}) = \text{ActorRNN}_i(o_{i,t}, h_{i,t}; \theta_i)
    \]
*   行动选择：根据概率分布 \(\pi_{\theta_i}\) 采样得到行动 \(a_{i,t} \sim \pi_{\theta_i}(\cdot | o_{i,t}, h_{i,t})\)。

### 3.2 循环价值网络 (Recurrent Critic Network)

*   价值函数 \(V_{\phi_i}\) 由参数为 \(\phi_i\) 的神经网络（包含 RNN 层）表示。
*   输入：当前观察 \(o_{i,t}\) 和 RNN 的上一个隐藏状态 \(h'_{i,t}\) (Critic 可以有独立的 RNN 状态)。
*   输出：下一个隐藏状态 \(h'_{i, t+1}\) 和对当前状态（观察和隐藏状态）下期望累积塑造后奖励的估计 \(V_{\phi_i}(o_{i,t}, h'_{i,t})\)。
    \[
    h'_{i,t+1}, V_{\phi_i}(o_{i,t}, h'_{i,t}) = \text{CriticRNN}_i(o_{i,t}, h'_{i,t}; \phi_i)
    \]
*   \(V_{\phi_i}(o_{i,t}, h'_{i,t}) \approx \mathbb{E} \left[ \sum_{k=t}^{T_{max}} \gamma^{k-t} r'_{i,k} | o_{i,t}, h'_{i,t} \right]\)，其中 \(\gamma\) 是折扣因子。

### 3.3 训练目标 (Training Objective - MAPPO)

MAPPO 通常最小化组合的 Actor 和 Critic 损失。

1.  **优势估计 (Advantage Estimation)**: 使用广义优势估计 (GAE) 或其他方法计算优势 \(A'_{i,t}\)，基于**塑造后奖励** \(r'_{i,t}\) 和价值估计 \(V_{\phi_i}\)。
    \[
    A'_{i,t} \approx \sum_{k=t}^{T_{max}} (\gamma \lambda)^{k-t} \delta'_{i,k}
    \]
    其中 \(\delta'_{i,k} = r'_{i,k} + \gamma V_{\phi_i}(o_{i, k+1}, h'_{i, k+1}) - V_{\phi_i}(o_{i,k}, h'_{i,k})\) 是 TD 误差，\(\lambda\) 是 GAE 参数。

2.  **Actor 损失 (PPO Clipped Objective)**:
    \[
    L^{CLIP}_i(\theta_i) = \mathbb{E}_t \left[ \min \left( \rho_t(\theta_i) A'_{i,t}, \text{clip}( \rho_t(\theta_i), 1-\epsilon, 1+\epsilon ) A'_{i,t} \right) \right]
    \]
    其中 \(\rho_t(\theta_i) = \frac{\pi_{\theta_i}(a_{i,t} | o_{i,t}, h_{i,t})}{\pi_{\theta_{i, \text{old}}}(a_{i,t} | o_{i,t}, h_{i,t})}\) 是重要性采样比率，\(\epsilon\) 是 PPO 的裁剪参数。目标是最大化 \(L^{CLIP}_i\)。

3.  **Critic 损失 (Value Function Loss)**:
    \[
    L^{VF}_i(\phi_i) = \mathbb{E}_t \left[ (V_{\phi_i}(o_{i,t}, h'_{i,t}) - V^{\text{target}}_{i,t})^2 \right]
    \]
    其中 \(V^{\text{target}}_{i,t} \approx \sum_{k=t}^{T_{max}} \gamma^{k-t} r'_{i,k}\) 是目标价值（例如，通过蒙特卡洛或 TD(λ) 计算的回报）。

4.  **（可选）熵损失 (Entropy Loss)**: 为了鼓励探索，通常会添加策略熵的正则化项 \(L^S_i(\theta_i) = \mathbb{E}_t [ H(\pi_{\theta_i}(\cdot | o_{i,t}, h_{i,t})) ]\)。

总损失通常是这些项的加权和。

## 4. 评估 (`evaluate.py`)

评估阶段使用训练好的策略网络 (\(\pi_{\theta_i}\)) 在环境中运行多个 episode，不进行梯度更新。主要目的是观察学习到的策略行为特征。

*   加载训练好的模型参数 \(\theta_i\)。
*   在环境中模拟智能体交互，根据 \(\pi_{\theta_i}\) 选择行动（通常是确定性地选择概率最高的行动，或进行采样）。
*   记录关键指标：每个智能体的价格 \(p_{i,t}\)、观察 \(o_{i,t}\)、销售额 \(S_{i,t}\)、库存 \(I_{i,t}\)、原始奖励 \(r_{i,t}\)、塑造后奖励 \(r'_{i,t}\) 等。
*   生成可视化图表，分析学习到的定价策略、库存动态和整体性能。

## 5. 总结

该系统通过 MAPPO 算法和 RNN 结构，使智能体能够在具有复杂动态（库存、提前期、缺货惩罚）的环境中学习定价策略。奖励塑造参数 \(\omega\) 允许我们调节智能体行为的合作程度，探索从纯粹自利到考虑集体利益的不同策略模式。数学形式化有助于清晰地理解系统的各个组成部分及其相互作用。
