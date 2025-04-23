# 项目的多智能体定价与库存管理模型：MDP 形式化

## 1. 引言

本项目模拟了一个包含 $N$ 个智能体（例如，零售商）的动态定价与库存管理系统。智能体需要在考虑需求波动、补货延迟、库存成本和缺货惩罚的情况下，学习最优的定价策略。系统引入了奖励塑造机制来影响智能体间的合作/竞争程度。我们将此系统形式化为一个多智能体马尔可夫决策过程 (MMDP)。

## 2. MDP/MMDP 组件定义

一个标准的 MDP 由元组 $(S, A, P, R, \gamma)$ 定义。在多智能体设置 (MMDP) 中，状态 $S$ 通常是全局状态，动作 $A$ 是所有智能体的联合动作空间。

### 2.1 状态空间 (State Space - $S$)

全局状态 $s_t \in S$ 在时间步 $t$ 包含了描述整个系统状况所需的所有信息。根据项目实现，状态 $s_t$ 可以定义为一个包含以下元素的元组：

$s_t = (\mathbf{I}_t, \mathcal{O}_t, \text{ShockState}_t, t)$

其中：

*   **库存水平 (Inventory Levels)**: $\mathbf{I}_t = (I_{0,t}, I_{1,t}, ..., I_{N-1,t})$
    *   $I_{i,t} \in [0, I_{max}]$ 是智能体 $i$ 在时间步 $t$ 开始时的库存量。$I_{max}$ 是最大库存容量 (`max_inventory_capacity`)。

*   **在途订单 (Pending Orders)**: $\mathcal{O}_t = \{ (\text{arrival\_step}_k, \text{agent\_id}_k, \text{amount}_k), ... \}$
    *   一个集合（或优先队列/heap），记录了所有已下单但尚未到达的补货信息。每个元组包含预计到达时间步、目标智能体 ID 和补货数量。

*   **需求冲击状态 (Shock State)**: $\text{ShockState}_t = (\text{is\_active}_t, \text{timer}_t, \text{target}_t)$
    *   $\text{is\_active}_t \in \{\text{True}, \text{False}\}$: 当前是否有冲击 (`shock_active`)。
    *   $\text{timer}_t \in \mathbb{N}_0$: 冲击剩余持续时间 (`shock_timer`)。
    *   $\text{target}_t \in \{-1, 0, ..., N-1\}$: 冲击影响的目标智能体 (`shock_target_agent`, -1 表示全部)。

*   **当前时间步 (Current Time Step)**: $t \in \{0, 1, ..., T_{max}-1\}$
    *   $T_{max}$ 是 episode 的最大长度 (`episode_length`)。

### 2.2 联合行动空间 (Joint Action Space - $A$)

联合行动空间是所有智能体个体行动空间的笛卡尔积。

*   **个体行动空间 (Individual Action Space)**: $\mathcal{A}_i = \{0, 1, ..., K-1\}$ for agent $i$.
    *   $K$ 是离散价格水平的数量 (`n_price_levels`)。动作 $a_{i,t} \in \mathcal{A}_i$ 是智能体 $i$ 在时间步 $t$ 选择的价格水平索引。

*   **联合行动 (Joint Action)**: $\mathbf{a}_t = (a_{0,t}, a_{1,t}, ..., a_{N-1,t}) \in A = \mathcal{A}_0 \times \mathcal{A}_1 \times ... \times \mathcal{A}_{N-1}$.

### 2.3 状态转移函数 (Transition Function - $P$)

状态转移函数 $P(s_{t+1} | s_t, \mathbf{a}_t)$ 定义了在状态 $s_t$ 执行联合行动 $\mathbf{a}_t$ 后，转移到下一个状态 $s_{t+1}$ 的概率分布。由于系统中存在多种随机性（需求噪声、冲击发生），这是一个概率性转移。其过程可以分解为以下步骤：

1.  **获取当前状态**: $s_t = (\mathbf{I}_t, \mathcal{O}_t, \text{ShockState}_t, t)$。
2.  **处理补货到达**:
    *   检查 $\mathcal{O}_t$，将所有满足 $\text{arrival\_step}_k \le t+1$ 的订单 $k$ 应用于对应智能体 $\text{agent\_id}_k$ 的库存 $I_{\text{agent\_id}_k, t}$（注意不超过 $I_{max}$）。
    *   更新在途订单集合 $\mathcal{O}_t \rightarrow \mathcal{O}'_t$（移除已到达的）。
    *   更新智能体的 `order_pending` 状态。
    *   得到补货后的库存 $\mathbf{I}'_t$。
3.  **更新冲击状态**:
    *   根据 $\text{ShockState}_t$ 和 `shock_prob` 概率，决定 $\text{ShockState}_{t+1}$（是否激活新冲击，或更新现有冲击的计时器）。这是**随机步骤**。
4.  **价格映射**:
    *   将联合行动（价格水平索引）$\mathbf{a}_t$ 映射为实际价格值 $\mathbf{p}_t = (p_{0,t}, ..., p_{N-1,t})$，其中 $p_{i,t} = \text{_price\_level\_to\_value}(a_{i,t}) = a_{i,t} + 1$。
5.  **计算潜在需求**:
    *   根据实际价格 $\mathbf{p}_t$、系统参数（基础需求、价格弹性、交叉弹性）和更新后的冲击状态 $\text{ShockState}_{t+1}$，计算每个智能体的潜在需求 $\mathbf{D}_t = (D_{0,t}, ..., D_{N-1,t})$。
    *   需求计算包含**随机噪声** (e.g., `np.random.normal`)。 $D_{i,t} = f_D(\mathbf{p}_t, \text{ShockState}_{t+1}, \text{Noise}_t)$。
6.  **计算实际销售与库存更新**:
    *   实际销售 $S_{i,t} = \max(0, \min(I'_{i,t}, D_{i,t}))$。
    *   期末库存 $I_{i, t+1} = I'_{i,t} - S_{i,t}$。
7.  **计算缺货**:
    *   计算未满足的需求 $U_{i,t} = \max(0, D_{i,t} - I'_{i,t})$。
8.  **触发新补货订单**:
    *   对于每个智能体 $i$，如果 $I_{i, t+1} < R_{th}$ (`replenishment_threshold`) 且智能体 $i$ 当前没有在途订单，则生成一个新订单 $(\text{arrival\_step} = t+1+L, i, R_{amt})$，并加入在途订单集合。
    *   更新在途订单集合 $\mathcal{O}'_t \rightarrow \mathcal{O}_{t+1}$。
    *   更新智能体的 `order_pending` 状态。
9.  **时间步进**: $t \rightarrow t+1$。
10. **形成下一状态**: $s_{t+1} = (\mathbf{I}_{t+1}, \mathcal{O}_{t+1}, \text{ShockState}_{t+1}, t+1)$。

由于步骤 3 和 5 的随机性，从 $(s_t, \mathbf{a}_t)$ 到 $s_{t+1}$ 的转移是概率性的。

### 2.4 奖励函数 (Reward Function - $R$)

奖励函数 $R(s_t, \mathbf{a}_t)$ 返回一个奖励向量 $\mathbf{r}'_t = (r'_{0,t}, ..., r'_{N-1,t})$，其中 $r'_{i,t}$ 是智能体 $i$ 在执行行动 $\mathbf{a}_t$ 后获得的**塑造后奖励**。计算过程如下：

1.  **计算中间量**（在状态转移过程中得到）：实际价格 $\mathbf{p}_t$、实际销售 $\mathbf{S}_t$、期末库存 $\mathbf{I}_{t+1}$、未满足需求 $\mathbf{U}_t$。
2.  **计算个体原始奖励** $r_{i,t}$：
    *   **利润 (Profit)**: $\text{Profit}_{i,t} = S_{i,t} \times (p_{i,t} - C_i)$ （$C_i$ 是单位成本 `costs[i]`）。
    *   **持有成本 (Holding Cost)**: $\text{HoldCost}_{i,t} = H \times I_{i, t+1}$ （$H$ 是单位持有成本 `holding_cost`）。
    *   **缺货惩罚 (Stockout Penalty)**: $\text{Penalty}_{i,t} = P_{so} \times U_{i,t}$ （$P_{so}$ 是单位缺货惩罚 `stockout_penalty_per_unit`）。
    *   **原始奖励**: $r_{i,t} = \text{Profit}_{i,t} - \text{HoldCost}_{i,t} - \text{Penalty}_{i,t}$
3.  **计算系统总奖励**:
    *   $R_{total, t} = \sum_{j=0}^{N-1} r_{j,t}$
4.  **计算塑造后奖励**:
    *   $r'_{i,t} = \omega \cdot r_{i,t} + (1 - \omega) \cdot R_{total, t}$
    *   其中 $\omega$ 是奖励塑造参数 (`omega`)。

最终的奖励函数输出为 $\mathbf{R}(s_t, \mathbf{a}_t) = (r'_{0,t}, r'_{1,t}, ..., r'_{N-1,t})$。

### 2.5 折扣因子 (Discount Factor - $\gamma$)

折扣因子 $\gamma \in [0, 1]$ 用于衡量未来奖励相对于当前奖励的重要性。在强化学习算法（如 PPO）的配置中通常会设定（例如，`gamma=0.99`）。

## 3. 目标 (Objective)

在 MMDP 框架下，目标是找到一个联合策略 $\pi(\mathbf{a}_t | s_t)$，使得每个智能体 $i$ 的期望累积折扣（塑造后）奖励最大化：

$$
\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{T_{max}-1} \gamma^t r'_{i,t} | s_0, \pi \right] \quad \forall i \in \{0, ..., N-1\}
$$

这个目标通常很难直接优化。

## 4. 与 Dec-POMDP 和实现的联系

*   **部分可观察性**: 实际上，每个智能体 $i$ 通常只能访问**局部观察** $o_{i,t}$ 而非完整的全局状态 $s_t$。局部观察 $o_{i,t}$ 由一个观察函数 $O(s_t, i)$ 生成，根据代码 `_get_obs()`，它通常包含：
    *   归一化的自身上一步价格 (或当前价格水平)。
    *   归一化的时间步。
    *   归一化的竞争对手平均价格。
    *   归一化的自身当前库存。
    $o_{i,t} = (\frac{a_{i, t-1}}{K-1}, \frac{t}{T_{max}}, \frac{\bar{a}_{\neg i, t-1}}{K-1}, \frac{I_{i,t}}{I_{init}})$ (近似表示)
    这使得问题更符合 **Dec-POMDP** 框架。

*   **智能体策略**: 智能体学习的是基于局部观察（和 RNN 隐藏状态 $h_{i,t}$）的**分散式策略** $\pi_i(a_{i,t} | o_{i,t}, h_{i,t})$。

*   **集中式训练 (CTDE)**: 像 MAPPO 这样的算法通常采用**集中式训练、分散式执行**。训练时，Critic 可能能访问全局状态或更多信息来改进值函数估计，但执行时 Actor 只依赖局部观察。

## 5. 总结

通过将项目形式化为 MMDP (并承认其 Dec-POMDP 的性质)，我们可以清晰地定义系统的状态、行动、动态和奖励结构。这种形式化有助于理解智能体面临的决策问题以及奖励塑造等机制如何影响其学习目标和最终行为。
