import sys
sys.path.append('../')

try:
    from pycore.tikzeng import *
except ImportError:
    print("ERROR: Could not import from pycore.tikzeng.")
    exit()

# Network parameters
STATE_DIM = 12
CRITIC_HIDDEN_DIM = 128

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # Input Layer (State) - using to_Conv to simulate a block
    to_Conv(name="input_state", offset="(0,0,0)", to="(0,0,0)",
            s_filer=str(STATE_DIM), n_filer=" ",
            width=4, height=STATE_DIM*2, depth=15, # Visual dimensions
            caption=f"State (batch, {STATE_DIM})"
            ),

    # FC1 Layer (followed by ReLU) - using to_Conv
    to_Conv(name="fc1_critic", offset="(3.5,0,0)", to="(input_state-east)",
            s_filer=str(CRITIC_HIDDEN_DIM), n_filer=" ",
            width=2, height=int(CRITIC_HIDDEN_DIM/4)+1, depth=int(CRITIC_HIDDEN_DIM/4)+1,
            caption=f"FC1 ({CRITIC_HIDDEN_DIM} units) + ReLU"
            ),

    # FC2 Layer (followed by ReLU) - using to_Conv
    to_Conv(name="fc2_critic", offset="(3,0,0)", to="(fc1_critic-east)",
            s_filer=str(CRITIC_HIDDEN_DIM), n_filer=" ",
            width=2, height=int(CRITIC_HIDDEN_DIM/4)+1, depth=int(CRITIC_HIDDEN_DIM/4)+1,
            caption=f"FC2 ({CRITIC_HIDDEN_DIM} units) + ReLU"
            ),

    # FC_Critic Layer (Output) - Value - using to_Conv
    to_Conv(name="output_value", offset="(3,0,0)", to="(fc2_critic-east)",
            s_filer="1", n_filer=" ", # Output is 1 neuron
            width=2, height=6, depth=6, # Make it visually distinct
            caption="Value (1 unit)"
            ),

    # Connections
    to_connection("input_state", "fc1_critic"),
    to_connection("fc1_critic", "fc2_critic"),
    to_connection("fc2_critic", "output_value"),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('/')[-1].split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"LaTeX file generated: {namefile + '.tex'}")

if __name__ == '__main__':
    main()
