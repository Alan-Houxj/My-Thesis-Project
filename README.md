**友情提示，虽然本人小菜鸡一枚东西也做的很一般，但是为了本论文设计的原创性得以保护，本项目仅作为毕业设计项目在2025/05/29-2025/05/30放出供老师检查我的毕业项目，过了这个时间我就把环境搭建和设计的网络结构等文件下掉了，只留下训练好的模型和评估文件供参考使用，如果后续有必要，可以随时联系我的邮箱，我将在24小时之内发送任何相关的源代码。非常欢迎任何一切的交流和批评，本菜鸡才入门强化学习不久，但是对这个领域非常感兴趣，大家的批评就是我最好的进步食粮，我会努力好好多学学相关知识的。** ❤️❤️❤️
**这是我的邮箱，欢迎大家来找我玩：Xiaojie_Hou2001@outlook.com**

# 基于 MAPPO 的多智能体动态定价与库存管理系统

## 项目简介

本项目实现了一个基于多智能体强化学习 (MARL) 的动态定价系统，其中每个智能体（代表一个电商产品）学习制定最优价格策略以最大化其回报。系统引入了库存管理机制，包括阈值补货、补货提前期、持有成本和缺货惩罚。此外，环境还模拟了随机需求冲击事件。智能体采用 MAPPO (Multi-Agent Proximal Policy Optimization) 算法进行训练，其中 Actor 网络包含 LSTM 层以处理时序信息。

## 主要功能

*   **多智能体强化学习**: 使用 MAPPO 算法训练多个智能体。
*   **动态定价**: 智能体学习在离散价格等级中选择最优价格。
*   **库存管理**:
    *   初始库存设置。
    *   基于阈值的补货策略。
    *   补货提前期模拟。
    *   库存持有成本。
    *   缺货惩罚。
    *   最大库存容量限制。
*   **需求动态性**:
    *   基础需求和价格弹性。
    *   交叉价格弹性（产品间的替代效应）。
    *   周期性需求波动和随机噪声。
    *   随机需求冲击事件（可配置概率、持续时间和影响幅度）。
*   **RNN Actor**: Actor 网络使用 LSTM 来处理观察序列，更好地捕捉时序依赖。
*   **中心化 Critic**: 使用共享的中心化 Critic 评估状态价值。
*   **奖励塑造**: 通过 `omega` 参数平衡个体奖励和系统总奖励。
*   **参数化环境**: 支持同质化和异质化智能体参数配置。
*   **训练与评估**:
    *   提供训练脚本 (`main.py`) 和评估脚本 (`evaluate.py`)。
    *   支持模型保存与加载。
    *   评估脚本包含与基线策略（固定低价、固定高价、随机）的对比。
    *   生成多种可视化图表，包括价格策略、个体详细指标（观察、销售、库存）、价格分布热力图和价格协调度。

## 文件结构

*   `environment.py`: 定义了动态定价和库存管理的多智能体环境。
*   `agent.py`: 实现了 MAPPO 智能体，包括 RNN Actor 和 Critic 网络结构及学习逻辑。
*   `main.py`: 主训练脚本，用于配置实验参数、初始化环境和智能体，并执行训练循环。
*   `evaluate.py`: 评估脚本，用于加载训练好的模型，在环境中运行评估周期，并与基线策略进行比较，生成可视化结果。
*   `README.md`: 本文件，提供项目说明。

## 环境要求与依赖

本项目基于 Python 实现，主要依赖以下库：

*   `torch`: 用于实现神经网络和强化学习算法。
*   `numpy`: 用于高效的数值计算。
*   `matplotlib`: 用于绘制训练和评估过程中的各种图表。
*   `seaborn`: 用于绘制美观的热力图。

建议使用 Anaconda 创建独立的 Python 环境来管理依赖。

```bash
conda create -n marl_pricing python=3.8
conda activate marl_pricing
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # 根据您的 CUDA 版本调整
pip install numpy matplotlib seaborn
```

## 使用方法

### 1. 训练模型

通过运行 `main.py` 脚本来训练智能体。可以调整命令行参数以配置实验。

**示例命令:**

```bash
python main.py --n_agents 3 --n_price_levels 5 --episode_len 200 --n_episodes 10000 \
               --actor_lr 1e-5 --critic_lr 5e-4 --rnn_hidden_dim 64 \
               --initial_inventory 500 --holding_cost 0.1 --stockout_penalty 0.5 \
               --replenishment_threshold 200 --replenishment_amount 100 --replenishment_lead_time 5 \
               --omega 0.8 --homogeneous \
               --save_dir ./results/exp_homogeneous_N3_penalty0.5 \
               --device cuda
```

**主要可配置参数 (`main.py`):**

*   `--n_agents`: 智能体数量。
*   `--n_price_levels`: 价格等级数量。
*   `--episode_len`: 每个周期的最大步数。
*   `--n_episodes`: 总训练周期数。
*   `--actor_lr`, `--critic_lr`: Actor 和 Critic 的学习率。
*   `--gamma`, `--gae_lambda`, `--clip_param`, `--entropy_coef`: PPO 算法超参数。
*   `--rnn_hidden_dim`, `--critic_hidden_dim`: Actor RNN 和 Critic 网络的隐藏层维度。
*   `--initial_inventory`, `--holding_cost`, `--stockout_penalty`: 库存相关成本参数。
*   `--replenishment_threshold`, `--replenishment_amount`, `--max_inventory_capacity`, `--replenishment_lead_time`: 补货策略参数。
*   `--shock_prob`, `--shock_duration`, `--shock_magnitude`: 需求冲击参数。
*   `--omega`: 奖励塑造参数 (0 到 1 之间)。
*   `--homogeneous`: 是否使用同质智能体参数 (设置此标志则激活)。
*   `--save_dir`: 模型和训练结果的保存目录。
*   `--device`: 训练设备 (`cpu` 或 `cuda`)。

训练完成后，模型参数（Actor 和 Critic 网络权重）以及训练曲线数据将保存在指定的 `--save_dir` 目录下。

### 2. 评估模型

使用 `evaluate.py` 脚本加载训练好的模型并进行评估。

**示例命令:**

```bash
python evaluate.py --model_dir ./results/exp_homogeneous_N3_penalty0.5 \
                   --n_agents 3 --n_price_levels 5 --episode_len 200 --homogeneous \
                   --initial_inventory 500 --holding_cost 0.1 --stockout_penalty 0.5 \
                   --target_agent_id 1 --eval_episodes 50 \
                   --device cuda
```

**主要可配置参数 (`evaluate.py`):**

*   `--model_dir`: 包含已训练模型的目录路径 (必需)。
*   `--save_dir`: 评估结果和图表的保存目录 (默认为 `--model_dir`)。
*   `--eval_episodes`: 用于平均结果的评估周期数。
*   `--target_agent_id`: 用于生成详细图表的目标智能体 ID。
*   其他环境参数应与训练时保持一致，以确保评估的有效性。

评估脚本会输出 MARL 模型与基线策略（固定低价、固定高价、随机策略）在平均系统回报上的对比。同时，会在 `--save_dir` (或 `--model_dir`) 下生成详细的评估图表。

## 输出结果

训练和评估过程会生成以下主要输出：

*   **模型文件**:
    *   `actor_i.pth`: 第 `i` 个智能体的 Actor 网络权重。
    *   `critic.pth`: 共享 Critic 网络的权重。
*   **训练数据 (NPY 文件)**:
    *   `avg_total_rewards.npy`: 训练过程中平滑后的平均总塑形奖励。
    *   `agent_i_avg_rewards.npy`: 第 `i` 个智能体平滑后的平均塑形奖励。
    *   `actor_i_loss.npy`: 第 `i` 个智能体的 Actor 损失历史。
    *   `critic_loss.npy`: Critic 损失历史。
*   **训练曲线图**:
    *   `training_curves_detailed_rnn.png`: 包含平滑奖励（总奖励和各智能体塑形奖励）以及损失曲线（Actor 和 Critic）的图表。
    *   `BASELINE/system_original_reward_raw_vs_smoothed.png`: 系统原始奖励（原始值 vs 平滑值）的图表。
    *   `BASELINE/agent_i_original_reward_raw_vs_smoothed.png`: 各智能体原始奖励（原始值 vs 平滑值）的图表。
*   **评估图表**:
    *   `evaluation_all_prices_rnn_inv.png`: 所有智能体在一个评估周期内的价格等级选择。
    *   `evaluation_agent<ID>_details_<STEPS>steps.png`: 目标智能体的详细指标，包括价格、归一化观察值（自身价格、时间、对手平均价格、库存）、实际销量和库存水平。
    *   `evaluation_price_distribution_heatmap.png`: 各智能体选择不同价格等级的频率分布热力图。
    *   `evaluation_price_coordination.png`: 价格协调度（跨智能体价格标准差）随时间变化的图。

## 注意事项

*   确保环境参数在训练和评估时保持一致，除非特定实验设计需要更改。
*   `omega` 参数对智能体的学习行为有显著影响。`omega=1` 表示纯个体奖励，`omega=0` 表示纯系统奖励（所有智能体获得相同的团队奖励）。
*   代码中的随机种子已固定，以保证实验的可复现性。

欢迎探索和扩展此项目！ 
