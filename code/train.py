import gymnasium as gym
import torch
import numpy as np
import wandb
from config import TDMPC_Config
from agent import TDMPCAgent
from buffer import ReplayBuffer

def train():
    # 1. Initialization
    cfg = TDMPC_Config()
    wandb.init(project="td-mpc-halfcheetah", name="fixed_agent_v2", config=cfg.__dict__)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TDMPCAgent(obs_dim, action_dim, cfg) 
    buffer = ReplayBuffer(1_000_000, obs_dim, action_dim) #Circular buffer for storing environment steps

    obs, _ = env.reset(seed=cfg.seed)
    episode_reward = 0
    episode_step = 0

    # Training was interrupted early due to early convergence (800k instead of 1M)
    for step in range(1, 1_000_001): 
        # For the first 5000 steps, we randomly fill the buffer
        if step < cfg.seed_steps:
            action = env.action_space.sample()
        else:
            # After collecting enough random experience, switch to MPPI planning
            action = agent.plan(obs, eval_mode=False, step=step, t0=(episode_step == 0))

        total_reward = 0
        for _ in range(cfg.action_repeat):
            next_obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        done = terminated or truncated

        scaled_reward = total_reward * 0.1   # scalling the reward
        episode_reward += total_reward
        buffer.add(obs, action, scaled_reward, next_obs, done)
        obs = next_obs
        episode_step += 1

        # After we gathered initial data, agent starts learning at every step
        if step >= cfg.seed_steps:
            loss_dict = agent.update(buffer, step)
            if step % 100 == 0:
                wandb.log(loss_dict, step=step)

        if done:
            wandb.log({"train/episode_reward": episode_reward}, step=step)
            print(f"Step {step} | Episode reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            episode_reward = 0
            episode_step = 0

    
    torch.save(agent.model.state_dict(), "tdmpc_cheetah_final.pt") 
    
if __name__ == "__main__":
    train()