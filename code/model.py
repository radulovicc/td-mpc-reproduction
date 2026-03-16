import torch
import torch.nn as nn
from config import TDMPC_Config

# Model initialization. If we initialized it randomly we would have problems with exploding or disappearing gradients.
def orthogonal_init(m):
    """Auxiliary function for orthogonal initialization (Deep RL standard)"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data) #At the beginning, we want our bias to be 0, so as not to introduce unnecessary bias at the start.
                                        #We want our output to initially depend only on the input x and the weight matrix W.

class TOLD(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg: TDMPC_Config):
        super().__init__()

        self.cfg = cfg
        h_dim = self.cfg.hidden_dim
        l_dim = self.cfg.latent_dim
        enc_dim = self.cfg.enc_hidden_dim

        # 1. ENCODER (h_theta)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, enc_dim),
            nn.ELU(),
            nn.Linear(enc_dim, l_dim)
        )
        # 2. DYNAMICS (d_theta) - Zk+1 = f(Zk, Uk) - a model that predicts the next latent representation based on the current and the action. This is the heart of the model, as it allows planning in latent space.
        self.dynamics = nn.Sequential(
            nn.Linear(l_dim + action_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, l_dim)
        )
        # 3. REWARD (R_theta) - the reward also depends on the state we are in at the moment and the action we take. This allows us to see which changes are beneficial to us and to plan actions that will bring us a greater reward.
        self.reward = nn.Sequential(
            nn.Linear(l_dim + action_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1)
        )
        # 4. VALUE (Q_theta)
        self.q_net1 = nn.Sequential(
            nn.Linear(l_dim + action_dim, h_dim),
            nn.LayerNorm(h_dim),  # It calms the state of the value ​​because it can often explode or disappear, and it also speeds up learning. Normalizes the vector.
                                  # This is very important because it normalizes the value so that it always has a mean of 0 and a std of 1, which means that our gradients will be more stable and that the value in ELU will always come from the same range.
                                  # The model learns much faster this way.
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1)
        )
        self.q_net2 = nn.Sequential(
            nn.Linear(l_dim + action_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1)
        )

        # 5. POLICY (pi_theta) - at the output we get a control signal for each of the 6 actuators. This signal tells us how much to move each of the 6 actuators to get the optimal action.
        self.policy = nn.Sequential(
            nn.Linear(l_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, action_dim),
            nn.Tanh() #We limit the output of the policy network to [-1, 1] because actions in most MuJoCo environments are limited to that range. This helps us get valid actions directly from the network without the need for additional scaling.
        )

        self.apply(orthogonal_init) #Goes through each network in the model and initializes it orthogonally
        nn.init.zeros_(self.reward[-1].weight)
        nn.init.zeros_(self.reward[-1].bias)
        
        nn.init.zeros_(self.q_net1[-1].weight)
        nn.init.zeros_(self.q_net1[-1].bias)
        nn.init.zeros_(self.q_net2[-1].weight)
        nn.init.zeros_(self.q_net2[-1].bias)
 
        #We set the weight and bias of the last layer of reward and value to 0, because we want the reward and value to be 0 at the beginning, so that we don't immediately give a big reward to the model for something it didn't do well at all. It must not be random.
    
    def forward(self, obs):
        # We pass through the encoder to get the latent representation
        z = self.encoder(obs)
        return z
    
if __name__ == "__main__":
    # We are making an instance of the Config
    konfiguracija = TDMPC_Config()

    
    model = TOLD(obs_dim=24, action_dim=6, cfg=konfiguracija)


    print(f"Model successfully made!")
    print(f"Latent dimension {model.cfg.latent_dim}")
    print(f"Hidden dimension: {model.cfg.hidden_dim}")