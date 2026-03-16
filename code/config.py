from dataclasses import dataclass

@dataclass
class TDMPC_Config:
    # Okruženje
    env_name: str = "HalfCheetah-v5"
    seed: int = 42
    seed_steps: int = 5000          # starting random steps

    # Arhitektura (TOLD model)
    latent_dim: int = 50
    hidden_dim: int = 512
    enc_hidden_dim: int = 256

    # MPPI / CEM planer
    horizon: int = 15                # H – longer horizon for cheetah
    iterations: int = 6               # J – number of CEM iterations
    num_samples: int = 512            # N – number of trajectories
    num_pi_samples: int = 25          # N_pi – number of policy trajectories
    elite_size: int = 64               # K – number of elites
    temperature: float = 0.5
    min_std: float = 0.01
    momentum: float = 0.1
    action_repeat: int = 2             

    # Trening
    batch_size: int = 512
    lr: float = 3e-4                    # Learning rate for the model without policy
    pi_lr: float = 3e-4                  # Learning rate for policy
    gamma: float = 0.99
    tau: float = 0.01                     # EMA coefficient
    update_freq: int = 2                   # update target every 2nd step
    grad_clip_norm: float = 20.0

    # Coefficients for losses
    rho: float = 0.5                       # temporal discount
    consistency_coef: float = 2.0  
    reward_coef: float = 0.5
    value_coef: float = 0.1

    # Linearni rasporedi (stringovi)
    std_schedule: str = "linear(1.0,0.1,100000)"   # starting std=1.0, ending std=0.1 after 100k steps
    horizon_schedule: str = "linear(5,15,100000)"   # we don't use it, but I'll leave it

    # Optional – prioritized replay
    per_alpha: float = 0.6
    per_beta: float = 0.4