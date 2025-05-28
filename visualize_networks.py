import torch
from torchviz import make_dot
import os

# --- 新增: 导入 HiddenLayer ---
import hiddenlayer as hl

# 假设 agent.py 与此脚本在同一目录或在 Python 路径中
from agent import Actor, Critic

# --- 配置参数 (基于项目中的默认值) ---
# Actor 参数
OBS_DIM = 4
ACTION_DIM = 5
RNN_HIDDEN_DIM = 64
FC_HIDDEN_DIM = 64 # Actor 内部的默认值

# Critic 参数
STATE_DIM = 12 # (obs_dim * n_agents = 4 * 3)
CRITIC_HIDDEN_DIM = 128

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出目录
OUTPUT_DIR = "network_visualizations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def visualize_actor():
    """可视化 Actor 网络 (使用 torchviz)"""
    print(f"--- Visualizing Actor Network (Torchviz) ---")
    print(f"Parameters: obs_dim={OBS_DIM}, action_dim={ACTION_DIM}, rnn_hidden_dim={RNN_HIDDEN_DIM}, fc_hidden_dim={FC_HIDDEN_DIM}")

    actor_model = Actor(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                        rnn_hidden_dim=RNN_HIDDEN_DIM, fc_hidden_dim=FC_HIDDEN_DIM).to(DEVICE)
    actor_model.eval() # 设置为评估模式

    # 为 Actor 创建虚拟输入
    # obs 形状: (batch, seq_len, obs_dim) 或 (batch, obs_dim)
    # 这里使用 (1, 1, obs_dim) 模拟单步单批次输入，LSTM 内部会处理
    dummy_obs = torch.randn(1, 1, OBS_DIM).to(DEVICE)
    # hidden_state (h_0, c_0)，每个形状 (num_layers*num_directions, batch, rnn_hidden_dim)
    # 对于 Actor 中的 LSTM (num_layers=1, num_directions=1)
    dummy_h0 = torch.randn(1, 1, RNN_HIDDEN_DIM).to(DEVICE)
    dummy_c0 = torch.randn(1, 1, RNN_HIDDEN_DIM).to(DEVICE)
    dummy_hidden_state = (dummy_h0, dummy_c0)

    try:
        # Actor 前向传播
        action_probs, _ = actor_model(dummy_obs, dummy_hidden_state)

        # 生成图
        # params 可以传入 actor_model.named_parameters() 来显示参数名和形状
        dot = make_dot(action_probs, params=dict(actor_model.named_parameters()),
                       show_attrs=True, show_saved=True)
        # --- 新增: 设置 DPI 来提高图像清晰度 ---
        dot.graph_attr['dpi'] = '300'
        # --- 修改结束 ---
        output_path = os.path.join(OUTPUT_DIR, "actor_network_torchviz")
        dot.render(output_path, format="png", cleanup=True)
        print(f"Actor network (torchviz) visualization saved to {os.path.abspath(output_path)}.png")

    except Exception as e:
        print(f"Error visualizing Actor network (torchviz): {e}")
        print("Make sure Graphviz is installed and in your system's PATH for torchviz.")

def visualize_critic():
    """可视化 Critic 网络 (使用 torchviz)"""
    print(f"--- Visualizing Critic Network (Torchviz) ---")
    print(f"Parameters: state_dim={STATE_DIM}, hidden_dim={CRITIC_HIDDEN_DIM}")

    critic_model = Critic(state_dim=STATE_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(DEVICE)
    critic_model.eval() # 设置为评估模式

    # 为 Critic 创建虚拟输入
    # state 形状: (batch, state_dim)
    dummy_state = torch.randn(1, STATE_DIM).to(DEVICE)

    try:
        # Critic 前向传播
        value = critic_model(dummy_state)

        # 生成图
        dot = make_dot(value, params=dict(critic_model.named_parameters()),
                       show_attrs=True, show_saved=True)
        # --- 新增: 设置 DPI 来提高图像清晰度 ---
        dot.graph_attr['dpi'] = '300'
        # --- 修改结束 ---
        output_path = os.path.join(OUTPUT_DIR, "critic_network_torchviz")
        dot.render(output_path, format="png", cleanup=True)
        print(f"Critic network (torchviz) visualization saved to {os.path.abspath(output_path)}.png")

    except Exception as e:
        print(f"Error visualizing Critic network (torchviz): {e}")
        print("Make sure Graphviz is installed and in your system's PATH for torchviz.")


# <---------------- BEGIN HIDDENLAYER FUNCTIONS ---------------->
def visualize_actor_hl():
    """可视化 Actor 网络 (使用 HiddenLayer)"""
    print(f"--- Visualizing Actor Network (HiddenLayer) ---")
    actor_model = Actor(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                        rnn_hidden_dim=RNN_HIDDEN_DIM, fc_hidden_dim=FC_HIDDEN_DIM).to(DEVICE)
    actor_model.eval()

    # HiddenLayer 的 build_graph 需要元组形式的输入参数
    dummy_obs = torch.randn(1, 1, OBS_DIM).to(DEVICE)
    dummy_h0 = torch.randn(1, 1, RNN_HIDDEN_DIM).to(DEVICE)
    dummy_c0 = torch.randn(1, 1, RNN_HIDDEN_DIM).to(DEVICE)
    dummy_hidden_state_tuple = (dummy_h0, dummy_c0)
    # 注意: 对于包含多个输入的模型，第二个参数应该是包含所有输入的元组
    graph_input_args = (dummy_obs, dummy_hidden_state_tuple)

    try:
        # 创建 HiddenLayer Graph 对象
        # hl.build_graph 的第二个参数是 *args 形式，所以我们直接传递元组中的元素
        graph = hl.build_graph(actor_model, graph_input_args)
        output_path = os.path.join(OUTPUT_DIR, "actor_network_hl")
        graph.save(output_path, format="png")
        print(f"Actor network (HiddenLayer) visualization saved to {os.path.abspath(output_path)}.png")
    except Exception as e:
        print(f"Error visualizing Actor network (HiddenLayer): {e}")
        print("Make sure Graphviz is installed and in your system's PATH for HiddenLayer.")
        print("You can install HiddenLayer using: pip install hiddenlayer")

def visualize_critic_hl():
    """可视化 Critic 网络 (使用 HiddenLayer)"""
    print(f"--- Visualizing Critic Network (HiddenLayer) ---")
    critic_model = Critic(state_dim=STATE_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(DEVICE)
    critic_model.eval()

    dummy_state = torch.randn(1, STATE_DIM).to(DEVICE)
    # 对于只有一个输入的模型，可以直接传递该输入
    graph_input_args = (dummy_state,)

    try:
        graph = hl.build_graph(critic_model, graph_input_args)
        output_path = os.path.join(OUTPUT_DIR, "critic_network_hl")
        graph.save(output_path, format="png")
        print(f"Critic network (HiddenLayer) visualization saved to {os.path.abspath(output_path)}.png")
    except Exception as e:
        print(f"Error visualizing Critic network (HiddenLayer): {e}")
        print("Make sure Graphviz is installed and in your system's PATH for HiddenLayer.")
        print("You can install HiddenLayer using: pip install hiddenlayer")
# <---------------- END HIDDENLAYER FUNCTIONS ---------------->

# <---------------- BEGIN ONNX EXPORT FUNCTIONS ---------------->
def export_actor_to_onnx():
    """将 Actor 网络导出为 ONNX 格式"""
    print(f"--- Exporting Actor Network to ONNX ---")
    actor_model = Actor(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                        rnn_hidden_dim=RNN_HIDDEN_DIM, fc_hidden_dim=FC_HIDDEN_DIM).to(DEVICE)
    actor_model.eval()

    dummy_obs = torch.randn(1, 1, OBS_DIM).to(DEVICE)
    dummy_h0 = torch.randn(1, 1, RNN_HIDDEN_DIM).to(DEVICE)
    dummy_c0 = torch.randn(1, 1, RNN_HIDDEN_DIM).to(DEVICE)
    dummy_hidden_state_tuple = (dummy_h0, dummy_c0)
    # ONNX export需要一个包含所有输入的元组
    # Actor的forward方法接收 obs 和一个包含 (h,c) 的元组
    # 所以传递给 export 的 args 应该是 (dummy_obs, dummy_hidden_state_tuple)
    # 然而，torch.onnx.export 的 args 参数期望的是一个扁平化的输入元组
    # 如果模型的 forward 方法是 def forward(self, obs, hidden_state) 并且 hidden_state 是一个元组 (h,c)
    # 那么 ONNX 导出时，需要将 hidden_state 解包传入。
    # 但由于 Actor 的 forward 明确接收 obs 和 hidden_state (作为一个元组参数)
    # 我们应该将它们作为两个独立的输入传递给 build_graph 或 onnx.export，或者调整模型使其接收扁平输入。
    # HiddenLayer 的 build_graph(model, (arg1, arg2)) 是正确的。
    # 对于 torch.onnx.export, 如果 forward 是 forward(self, obs, hidden_tuple), 那么 args=(obs_tensor, hidden_tuple_tensor) 应该是可以的。
    # 但通常为了更清晰的 ONNX 图，模型输入最好是扁平的张量。

    # 我们先尝试直接传递元组作为第二个参数，如果不行再调整。
    # 根据 Actor 的 forward(self, obs, hidden_state) 签名，其中 hidden_state 是 (h,c)
    # args 应该是 (dummy_obs, dummy_hidden_state_tuple)
    # 实际上，torch.onnx.export 的 args 参数需要一个扁平化的元组。
    # 如果 forward(self, obs, hidden_state_tuple), 其中 hidden_state_tuple = (h,c)
    # 那么 args 应该是 (dummy_obs, (dummy_h0, dummy_c0)) 
    # 不，args 应该是 (dummy_obs, dummy_h0, dummy_c0) 如果 forward 被定义为 forward(self, obs, h, c)
    # 既然 forward 是 forward(self, obs, hidden_state) 而 hidden_state 是 (h,c)元组，
    # onnx export 通常需要将元组输入也视为单个输入（如果它是模型的一个参数）。
    # 但为了更好的 ONNX 图和兼容性，模型通常期望扁平化的输入。

    # 检查 Actor 的 forward 方法：forward(self, obs, hidden_state)
    # hidden_state 本身是一个元组 (h_0, c_0)。
    # 所以，当调用 actor_model(dummy_obs, dummy_hidden_state_tuple) 时，是正确的。
    # 对于 torch.onnx.export，第二个参数 args 也应该是这样一个元组：
    # args = (dummy_obs, dummy_hidden_state_tuple) 是不正确的，因为 hidden_state_tuple 会被视为一个单独的非张量输入。
    # 正确的方式是，如果模型的 forward 方法接收的是 (obs, h, c)，那么 args=(dummy_obs, dummy_h0, dummy_c0)。
    # 如果模型的 forward 方法接收的是 (obs, hidden_tuple)，那么通常需要修改模型或使用包装器。
    # 为了简单起见，我们先假设 HiddenLayer 和 torchviz 的处理方式对于 ONNX 也是类似的，即它能处理元组形式的 hidden_state。
    # **修正**：torch.onnx.export 的 `args` 参数应该是一个包含模型所有输入的元组，且这些输入都应该是张量。
    # 如果模型的 forward 是 `def forward(self, obs, hidden_state)` 并且 `hidden_state` 是一个元组 `(h,c)`，
    # 那么在导出时，我们需要将 `h` 和 `c` 作为单独的输入提供给 `torch.onnx.export`。
    # 这意味着我们需要修改 Actor 的 forward 方法以接收扁平化的 `h` 和 `c`，或者创建一个包装模型。

    # 为了避免修改原始模型，我们将创建一个简单的包装器供ONNX导出
    class ActorOnnxWrapper(torch.nn.Module):
        def __init__(self, actor_model):
            super().__init__()
            self.actor_model = actor_model
        def forward(self, obs, h_0, c_0):
            return self.actor_model(obs, (h_0, c_0))

    actor_wrapper = ActorOnnxWrapper(actor_model).to(DEVICE)
    actor_wrapper.eval()

    onnx_input_args = (dummy_obs, dummy_h0, dummy_c0)
    onnx_output_path = os.path.join(OUTPUT_DIR, "actor_network.onnx")

    try:
        torch.onnx.export(actor_wrapper,
                          onnx_input_args,
                          onnx_output_path,
                          input_names=['obs', 'h_in', 'c_in'],
                          output_names=['action_probs', 'h_out', 'c_out'], # Actor的包装器会返回两个输出
                          dynamic_axes={ # 如果批处理大小或序列长度可变
                              'obs': {0: 'batch_size', 1: 'seq_len'},
                              'h_in': {1: 'batch_size'},
                              'c_in': {1: 'batch_size'},
                              'action_probs': {0: 'batch_seq'},
                              'h_out': {1: 'batch_size'},
                              'c_out': {1: 'batch_size'}
                          },
                          opset_version=11) # 一个常用的稳定版本
        print(f"Actor network exported to ONNX: {os.path.abspath(onnx_output_path)}")
    except Exception as e:
        print(f"Error exporting Actor network to ONNX: {e}")

def export_critic_to_onnx():
    """将 Critic 网络导出为 ONNX 格式"""
    print(f"--- Exporting Critic Network to ONNX ---")
    critic_model = Critic(state_dim=STATE_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(DEVICE)
    critic_model.eval()

    dummy_state = torch.randn(1, STATE_DIM).to(DEVICE)
    onnx_input_args = (dummy_state,)
    onnx_output_path = os.path.join(OUTPUT_DIR, "critic_network.onnx")

    try:
        torch.onnx.export(critic_model,
                          onnx_input_args,
                          onnx_output_path,
                          input_names=['state'],
                          output_names=['value'],
                          dynamic_axes={ # 如果批处理大小可变
                              'state': {0: 'batch_size'},
                              'value': {0: 'batch_size'}
                          },
                          opset_version=11)
        print(f"Critic network exported to ONNX: {os.path.abspath(onnx_output_path)}")
    except Exception as e:
        print(f"Error exporting Critic network to ONNX: {e}")

# <---------------- END ONNX EXPORT FUNCTIONS ---------------->

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # --- Torchviz Visualizations ---
    # visualize_actor() # 取消注释以生成 torchviz 图
    # print("-" * 30)
    # visualize_critic() # 取消注释以生成 torchviz 图
    # print("=" * 40)

    # --- HiddenLayer Visualizations (仍会尝试，但已知可能失败) ---
    visualize_actor_hl()
    print("-" * 30)
    visualize_critic_hl()
    print("=" * 40)

    # --- ONNX Export ---
    export_actor_to_onnx()
    print("-" * 30)
    export_critic_to_onnx()

    print(f"\nAll visualizations and ONNX models saved in directory: {os.path.abspath(OUTPUT_DIR)}")

# 提示:
# 1. 确保 agent.py 文件与此脚本在同一目录下，或者 agent.py 所在的目录在 PYTHONPATH 中。
# 2. 运行此脚本前，请确保已安装所需的库:
#    - Torchviz: conda install -c conda-forge python-graphviz torchviz
#    - HiddenLayer: pip install hiddenlayer (确保 Graphviz 也已安装并配置)
# 3. 生成的图像将保存在名为 'network_visualizations' 的子目录中。 