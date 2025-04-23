import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# --- 辅助函数 ---
def init_weights(m):
    """初始化网络权重。"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(1.0) # 可能需要调整缩放
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    # LSTM 权重初始化可以更复杂，这里暂时省略，使用默认
    elif isinstance(m, nn.LSTM):
         for name, param in m.named_parameters():
             if 'bias' in name:
                 nn.init.constant_(param, 0.0)
             elif 'weight_ih' in name:
                 nn.init.orthogonal_(param)
                 param.data.mul_(1.0)
             elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
                 param.data.mul_(1.0)

# --- Actor 网络 ---
class Actor(nn.Module):
    """
    MAPPO 的 Actor 网络，使用 LSTM 处理时序信息。
    输入: (局部观察 obs, 上一隐藏状态 hidden_state)
    输出: (动作概率分布, 下一隐藏状态 next_hidden_state)
    """
    def __init__(self, obs_dim, action_dim, rnn_hidden_dim=64, fc_hidden_dim=64):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_dim = action_dim

        # RNN 层 (这里用 LSTM)
        self.lstm = nn.LSTM(obs_dim, rnn_hidden_dim, batch_first=True) # batch_first=True 输入形状 (batch, seq, feature)

        # 后续全连接层
        self.fc1 = nn.Linear(rnn_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self, obs, hidden_state):
        """
        Args:
            obs (torch.Tensor): 输入观察，形状 (batch, seq_len, obs_dim) 或 (batch, obs_dim) if seq_len=1
            hidden_state (tuple): (h_0, c_0)，每个形状 (1, batch, rnn_hidden_dim)

        Returns:
            tuple: (action_probs, next_hidden_state)
                   - action_probs: 动作概率，形状 (batch * seq_len, action_dim) 或 (batch, action_dim)
                   - next_hidden_state: (h_n, c_n)，更新后的隐藏状态
        """
        # LSTM 输入期望形状 (batch, seq_len, input_size)
        # 如果输入是单个时间步，需要增加 seq_len 维度
        if obs.ndim == 2:
            obs = obs.unsqueeze(1) # (batch, 1, obs_dim)

        # LSTM 输出: output (batch, seq_len, rnn_hidden_dim), (h_n, c_n)
        lstm_out, next_hidden_state = self.lstm(obs, hidden_state)

        # 将 LSTM 输出喂给后续 FC 层
        # 需要调整形状为 (batch * seq_len, rnn_hidden_dim)
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.rnn_hidden_dim)

        x = self.relu(self.fc1(lstm_out_reshaped))
        action_logits = self.fc2(x)
        action_probs = self.softmax(action_logits)

        # 如果输入是单个时间步，输出也应对应调整形状
        if obs.shape[1] == 1:
            action_probs = action_probs.view(-1, self.action_dim) # (batch, action_dim)

        return action_probs, next_hidden_state

# --- Critic 网络 ---
class Critic(nn.Module):
    """
    MAPPO 的中心化 Critic 网络。
    输入: 全局状态 state
    输出: 状态价值 V(s)
    """
    def __init__(self, state_dim, hidden_dim=128): # Critic 通常比 Actor 更大
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # 输出单个价值
        self.relu = nn.ReLU()
        self.apply(init_weights) # 应用权重初始化

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# --- MAPPO Agent ---
class MAPPOAgent:
    """
    管理 RNN Actor 和 Critic 网络，并执行 MAPPO 更新。
    """
    def __init__(self, obs_dim, state_dim, action_dim, n_agents, agent_id,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,gae_lambda=0.95,
                 clip_param=0.2, entropy_coef=0.01, update_epochs=10,
                 rnn_hidden_dim=64, device='cpu'): # 添加 rnn_hidden_dim
        """
        初始化 MAPPO 代理逻辑 (RNN 版本)。

        Args:
            obs_dim (int): 单个智能体的观察维度。
            state_dim (int): 全局状态维度。
            action_dim (int): 单个智能体的动作维度（可选价格数量）。
            n_agents (int): 智能体总数。
            agent_id (int): 当前智能体的 ID (用于区分 Actor)。
            actor_lr (float): Actor 学习率。
            critic_lr (float): Critic 学习率。
            gamma (float): 折扣因子。
            gae_lambda (float): GAE lambda 参数。
            clip_param (float): PPO 裁剪参数 epsilon。
            entropy_coef (float): 熵正则化系数。
            update_epochs (int): 每次学习迭代的更新次数。
            rnn_hidden_dim (int): RNN 隐藏层维度。
            device (str): 'cpu' 或 'cuda'。
        """
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.agent_id = agent_id # 标识符，虽然网络结构相同
        self.device = device

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.rnn_hidden_dim = rnn_hidden_dim # RNN 隐藏层维度

        # --- 网络 ---
        self.actor = Actor(obs_dim, action_dim, rnn_hidden_dim=self.rnn_hidden_dim).to(device)
        # 所有智能体共享一个 Critic (在训练脚本中实例化一次即可)
        # 这里为了结构完整先放一个，但优化器需要在外部共享 Critic 时处理
        # 或者传入共享的 Critic 和优化器
        # self.critic = Critic(state_dim).to(device) # 暂时注释，将在主训练脚本创建共享 Critic

        # --- 优化器 ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr) # Critic 优化器也应共享

        print(f"初始化 MAPPO (RNN) 智能体 {self.agent_id} (Actor)")
        # print(f"  Actor 网络结构: {self.actor}") # 取消注释以查看结构
        # print(f"  Critic 网络结构 (共享): {self.critic}")

    def init_hidden(self, batch_size=1):
        """初始化 Actor 的隐藏状态 (h_0, c_0)"""
        # LSTM 的 hidden state 是一个元组 (h, c)
        # 形状: (num_layers * num_directions, batch, hidden_size)
        # 这里 num_layers=1, num_directions=1
        h0 = torch.zeros(1, batch_size, self.rnn_hidden_dim).to(self.device)
        c0 = torch.zeros(1, batch_size, self.rnn_hidden_dim).to(self.device)
        return (h0, c0)

    def choose_action(self, obs, hidden_state, deterministic=False):
        """
        根据当前观察和隐藏状态选择动作。

        Args:
            obs (np.ndarray): 当前智能体的局部观察 (obs_dim,)
            hidden_state (tuple): 上一个时间步的隐藏状态 (h, c)。
            deterministic (bool): 是否选择确定性动作。

        Returns:
            tuple: (action, action_log_prob, next_hidden_state)
                   - action (int): 选择的动作索引。
                   - action_log_prob (torch.Tensor): 该动作的对数概率。
                   - next_hidden_state (tuple): 更新后的隐藏状态 (h, c)。
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device) # (1, obs_dim)
        self.actor.eval()
        with torch.no_grad():
            # Actor 需要 obs 和 hidden_state
            action_probs, next_hidden_state = self.actor(obs, hidden_state) # 输入单步 obs，形状 (1, 1, obs_dim) 内部处理
        self.actor.train()

        dist = Categorical(action_probs) # probs 形状 (1, action_dim)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        action_log_prob = dist.log_prob(action)

        # next_hidden_state 需要 detach 以便下次使用
        next_hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())

        return action.item(), action_log_prob.cpu(), next_hidden_state

    # 注意：learn 方法通常在收集完一批数据后，在主训练脚本中调用
    # 它需要访问所有智能体的轨迹数据和共享的 Critic
    # 这里只是一个结构示例，实际实现可能在 main.py 中更合适

    def _compute_advantages(self, rewards, values, dones, next_values):
        """计算 GAE 优势。"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            # GAE 计算中 dones 需要处理好边界和类型
            is_done = dones[t].item() > 0.5 # 确保是布尔或 0/1
            mask = 1.0 - float(is_done)

            if is_done:
                delta = rewards[t] - values[t]
                last_gae_lam = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * mask # 使用 mask
            advantages[t] = last_gae_lam
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # --- learn 方法需要重大修改以处理序列和隐藏状态 ---
    def learn(self, batch, shared_critic, shared_critic_optimizer):
        """
        使用收集到的经验批次更新 Actor 和共享 Critic 网络 (RNN 版本)。

        Args:
            batch (dict): 包含轨迹数据的字典，现在需要包含隐藏状态：
                          'obs': (batch_size, ep_len, n_agents, obs_dim)
                          'state': (batch_size, ep_len, state_dim)
                          'actions': (batch_size, ep_len, n_agents)
                          'rewards': (batch_size, ep_len, n_agents)
                          'dones': (batch_size, ep_len, n_agents)
                          'old_log_probs': (batch_size, ep_len, n_agents)
                          'actor_h_in': (batch_size, ep_len, n_agents, rnn_hidden_dim) # 输入 LSTM 的 h
                          'actor_c_in': (batch_size, ep_len, n_agents, rnn_hidden_dim) # 输入 LSTM 的 c
                          'next_state': (batch_size, ep_len, state_dim) # 假设存在
                          'masks': (batch_size, ep_len, n_agents) # 用于处理 padded data

        Returns:
            tuple: (actor_loss, critic_loss) 平均损失值。
        """
        batch_size = batch['obs'].shape[0]
        ep_len = batch['obs'].shape[1]
        seq_len = ep_len # 假设不分块，直接用整个 episode

        # --- 数据准备 ---
        # 提取当前智能体的数据，并调整形状
        # (batch_size, seq_len, feature_dim) -> (batch_size * seq_len, feature_dim) 通常不适用于 RNN
        # RNN 处理需要保持序列维度: (batch_size, seq_len, feature_dim)
        obs_batch_seq = torch.FloatTensor(batch['obs'][:, :, self.agent_id]).to(self.device) # (bs, sl, obs_dim)
        state_batch_seq = torch.FloatTensor(batch['state']).to(self.device) # (bs, sl, state_dim)
        action_batch_seq = torch.LongTensor(batch['actions'][:, :, self.agent_id]).unsqueeze(-1).to(self.device) # (bs, sl, 1)
        reward_batch_seq = torch.FloatTensor(batch['rewards'][:, :, self.agent_id]).unsqueeze(-1).to(self.device) # (bs, sl, 1)
        # dones 需要明确类型和形状用于 GAE
        done_batch_seq = torch.FloatTensor(batch['dones'][:, :, self.agent_id]).unsqueeze(-1).to(self.device) # (bs, sl, 1), 使用 float
        old_log_prob_batch_seq = torch.FloatTensor(batch['old_log_probs'][:, :, self.agent_id]).unsqueeze(-1).to(self.device) # (bs, sl, 1)
        # 获取初始隐藏状态 (序列开始时的状态)
        # 注意: batch 数据构造时需要确保 h_in/c_in 的第一步是有效的初始状态
        h_in = torch.FloatTensor(batch['actor_h_in'][:, 0, self.agent_id]).unsqueeze(0).to(self.device) # (1, bs, hidden_dim)
        c_in = torch.FloatTensor(batch['actor_c_in'][:, 0, self.agent_id]).unsqueeze(0).to(self.device) # (1, bs, hidden_dim)
        initial_hidden_state = (h_in, c_in)

        # 处理 masks (用于处理 padding, 如果 episode 长度不一)
        # 假设 'masks' 存在, 形状 (bs, sl, 1) 或 (bs, sl)
        masks = torch.FloatTensor(batch.get('masks', torch.ones_like(done_batch_seq))).to(self.device)
        if masks.ndim == 2: masks = masks.unsqueeze(-1) # 确保形状 (bs, sl, 1)

        # --- GAE and Returns Calculation ---
        with torch.no_grad():
            # Critic evaluates sequences of global states
            values_seq = shared_critic(state_batch_seq.view(-1, self.state_dim)).view(batch_size, seq_len, 1)
            next_state_batch_seq = torch.FloatTensor(batch['next_state']).to(self.device) # (bs, sl, state_dim)
            next_values_seq = shared_critic(next_state_batch_seq.view(-1, self.state_dim)).view(batch_size, seq_len, 1)

            # Calculate target returns for the Critic using CENTRALIZED rewards
            central_reward_batch_seq = torch.sum(torch.FloatTensor(batch['rewards']), dim=2).unsqueeze(-1).to(self.device) # Sum rewards across agents -> (bs, sl, 1)
            # Use GLOBAL done signal for critic target calculation (episode ends for value when all are done)
            global_done_batch_seq = torch.all(torch.BoolTensor(batch['dones']), dim=2).unsqueeze(-1).float().to(self.device) # (bs, sl, 1)

            critic_target_returns_list = []
            for b in range(batch_size):
                # Ensure dones are float/long for GAE calculation
                 _, ret = self._compute_advantages(central_reward_batch_seq[b], values_seq[b], global_done_batch_seq[b], next_values_seq[b])
                 critic_target_returns_list.append(ret)
            critic_target_returns = torch.cat(critic_target_returns_list) # (bs * sl, 1)

            # Calculate advantages for the Actor using INDIVIDUAL rewards and dones
            # This reflects the individual agent's contribution
            actor_advantages_list = []
            # Individual dones
            individual_done_batch_seq = torch.FloatTensor(batch['dones'][:, :, self.agent_id]).unsqueeze(-1).to(self.device) # (bs, sl, 1)

            for b in range(batch_size):
                # Use individual rewards but global values for advantage calculation
                 adv, _ = self._compute_advantages(reward_batch_seq[b], values_seq[b], individual_done_batch_seq[b], next_values_seq[b])
                 actor_advantages_list.append(adv)
            actor_advantages = torch.cat(actor_advantages_list) # (bs * sl, 1)


        # Load and reshape masks (now based on timesteps)
        # batch['masks'] has shape (bs, sl, 1) after main.py change
        masks_flat = torch.FloatTensor(batch['masks']).view(-1, 1).to(self.device) # (bs * sl, 1)

        # --- 多轮 PPO 更新 ---
        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.update_epochs):
            # --- Critic 更新 ---
            # Critic 输入需要扁平化
            current_values = shared_critic(state_batch_seq.view(-1, self.state_dim)) # (bs * sl, 1)
            # 使用 critic_target_returns 和 masks_flat
            critic_loss = (((current_values - critic_target_returns)**2) * masks_flat).sum() / masks_flat.sum() # masked MSE

            shared_critic_optimizer.zero_grad()
            critic_loss.backward()
            # 修改：启用梯度裁剪
            nn.utils.clip_grad_norm_(shared_critic.parameters(), 0.5) # 限制梯度范数
            shared_critic_optimizer.step()

            # --- Actor 更新 ---
            # Actor 前向传播
            action_probs_seq, _ = self.actor(obs_batch_seq, initial_hidden_state) # (bs * sl, action_dim)
            dist = Categorical(action_probs_seq)
            new_log_probs = dist.log_prob(action_batch_seq.view(-1)).unsqueeze(-1) # (bs * sl, 1)
            entropy = dist.entropy().mean()

            # PPO 裁剪目标，使用 actor_advantages
            ratio = torch.exp(new_log_probs - old_log_prob_batch_seq.view(-1, 1))
            surr1 = ratio * actor_advantages # Use actor-specific advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * actor_advantages

            # 应用 mask (基于时间步有效性)
            masked_min_surr = torch.min(surr1, surr2) * masks_flat
            actor_loss = -(masked_min_surr.sum() / masks_flat.sum()) - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 修改：启用梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 限制梯度范数
            self.actor_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return total_actor_loss / self.update_epochs, total_critic_loss / self.update_epochs

# --- 示例 (仅结构，不运行) ---
if __name__ == '__main__':
    # 假设的环境信息
    env_info = {"n_agents": 3, "obs_shape": 3, "state_shape": 9, "n_actions": 5, "episode_limit": 50}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_hidden_dim = 64 # RNN 隐藏层大小

    # 在主脚本中创建共享 Critic
    shared_critic_net = Critic(env_info['state_shape']).to(device)
    shared_critic_opt = optim.Adam(shared_critic_net.parameters(), lr=1e-3)

    # 为每个智能体创建 Actor 逻辑 (网络实例不同，但共享 Critic 实例)
    agents = []
    for i in range(env_info['n_agents']):
        agent = MAPPOAgent(
            obs_dim=env_info['obs_shape'],
            state_dim=env_info['state_shape'],
            action_dim=env_info['n_actions'],
            n_agents=env_info['n_agents'],
            agent_id=i,
            rnn_hidden_dim=rnn_hidden_dim, # 传入 RNN hidden dim
            device=device
            # 其他超参数...
        )
        agents.append(agent)

    # 模拟一个智能体的动作选择
    dummy_obs = np.random.rand(env_info['obs_shape'])
    hidden_state = agents[0].init_hidden() # 获取初始隐藏状态
    action, log_prob, next_hidden_state = agents[0].choose_action(dummy_obs, hidden_state)
    print(f"Agent 0 选择动作: {action}, Log Prob: {log_prob}")
    print(f"Agent 0 下一隐藏状态 h shape: {next_hidden_state[0].shape}, c shape: {next_hidden_state[1].shape}")

    # 模拟学习过程 (需要构造 batch 数据)
    # batch = ... # 从经验回放缓冲区获取
    # actor_loss, critic_loss = agents[0].learn(batch, shared_critic_net, shared_critic_opt)
    # print(f"Agent 0 学习损失 - Actor: {actor_loss}, Critic: {critic_loss}")

    print("\nagent.py (RNN version) 文件结构创建完成。")
    print("注意: learn 方法的实现依赖于 main.py 中正确构造包含隐藏状态的批处理数据。")
