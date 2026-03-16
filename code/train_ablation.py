import gymnasium as gym
import torch
import numpy as np
import wandb
from config import TDMPC_Config
from agent import TDMPCAgent
from buffer import ReplayBuffer

def train_ablation():
    cfg = TDMPC_Config()
    cfg.consistency_coef = 0.0   # Turning off the consistency loss
    cfg.env_name = "HalfCheetah-v5"  

    wandb.init(project="td-mpc-halfcheetah", name="halfcheetah_ablation_no_consistency_800k", config=cfg.__dict__)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TDMPCAgent(obs_dim, action_dim, cfg)
    buffer = ReplayBuffer(1000000, obs_dim, action_dim)

    obs, _ = env.reset(seed=cfg.seed)
    episode_reward = 0
    episode_step = 0

    for step in range(1, 800_001):  
        if step < cfg.seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.plan(obs, eval_mode=False, step=step, t0=(episode_step == 0))

        total_reward = 0
        for _ in range(cfg.action_repeat):
            next_obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        done = terminated or truncated

        scaled_reward = total_reward * 0.1
        buffer.add(obs, action, scaled_reward, next_obs, done)
        obs = next_obs
        episode_reward += total_reward
        episode_step += 1

        if step >= cfg.seed_steps:
            loss_dict = agent.update(buffer, step)
            if step % 100 == 0:
                wandb.log(loss_dict, step=step)

        if done:
            wandb.log({"train/episode_reward": episode_reward}, step=step)
            print(f"Step {step} | Ep reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            episode_reward = 0
            episode_step = 0

    # Saving the model after the training
    torch.save(agent.model.state_dict(), "tdmpc_cheetah_ablation_v2.pt")
    wandb.finish()

if __name__ == "__main__":
    train_ablation()