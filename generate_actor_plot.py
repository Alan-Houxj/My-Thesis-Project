import sys
sys.path.append('../')

try:
    from pycore.tikzeng import *
except ImportError:
    print("ERROR: Could not import from pycore.tikzeng.")
    exit()

# Network parameters
OBS_DIM = 4
ACTION_DIM = 5
RNN_HIDDEN_DIM = 64
FC_HIDDEN_DIM = 64

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # Input Layer (Observation) - using to_Conv to simulate a block
    to_Conv(name="input_obs", offset="(0,0,0)", to="(0,0,0)",
            s_filer=str(OBS_DIM), n_filer=" ",
            width=4, height=OBS_DIM*4, depth=20,
            caption=f"Observation (batch, seq, {OBS_DIM})"
            ),

    # LSTM Layer - using to_Conv to simulate a block
    to_Conv(name="lstm_layer", offset="(3.5,0,0)", to="(input_obs-east)",
            s_filer=str(RNN_HIDDEN_DIM), n_filer="LSTM",
            width=10, height=int(RNN_HIDDEN_DIM/2)+1, depth=int(RNN_HIDDEN_DIM/2)+1,
            caption=f"Out: (batch, seq, {RNN_HIDDEN_DIM})"
            ),

    # FC1 Layer (followed by ReLU, noted in caption) - using to_Conv
    to_Conv(name="fc1", offset="(4,0,0)", to="(lstm_layer-east)",
            s_filer=str(FC_HIDDEN_DIM), n_filer=" ",
            width=2, height=int(FC_HIDDEN_DIM/2)+1, depth=int(FC_HIDDEN_DIM/2)+1,
            caption=f"FC1 ({FC_HIDDEN_DIM} units) + ReLU"
            ),

    # FC_Actor Layer (Output, followed by Softmax) - using to_Conv
    to_Conv(name="fc_actor", offset="(3,0,0)", to="(fc1-east)",
            s_filer=str(ACTION_DIM), n_filer=" ",
            width=2, height=ACTION_DIM*4, depth=ACTION_DIM*4,
            caption=f"FC Actor ({ACTION_DIM} units)"
            ),
    to_SoftMax(name="softmax_actor", s_filer=str(ACTION_DIM), offset="(0.5,0,0)", to="(fc_actor-east)",
               width=3, height=ACTION_DIM*4, depth=3, caption="Softmax"),

    # Connections
    to_connection("input_obs", "lstm_layer"),
    to_connection("lstm_layer", "fc1"),
    to_connection("fc1", "fc_actor"),
    to_connection("fc_actor", "softmax_actor"),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('/')[-1].split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"LaTeX file generated: {namefile + '.tex'}")

if __name__ == '__main__':
    main()
