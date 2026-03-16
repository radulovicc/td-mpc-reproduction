import torch
import torch.nn as nn
import numpy as np
from model import TOLD
from config import TDMPC_Config
from utils import linear_schedule, ema

class TDMPCAgent:
    def __init__(self, obs_dim, action_dim, cfg: TDMPC_Config):
        self.cfg = cfg
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Agent's brain - Model and target network
        self.model = TOLD(obs_dim, action_dim, cfg).to(self.device)
        self.model_target = TOLD(obs_dim, action_dim, cfg).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()
        
        # 2. Two separated optimizators
        model_params = list(self.model.encoder.parameters()) + \
                       list(self.model.dynamics.parameters()) + \
                       list(self.model.reward.parameters()) + \
                       list(self.model.q_net1.parameters()) + \
                       list(self.model.q_net2.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=self.cfg.lr)
        self.pi_optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=self.cfg.pi_lr)
        
        # 3. State for planning - memory for MPC
        self.plan_mu = torch.zeros(self.cfg.horizon, self.action_dim, device=self.device)
        self._prev_mean = None
        self.std = float(linear_schedule(self.cfg.std_schedule, 0))

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=0, t0=True):
        """
        Planira akciju koristeći CEM.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z0 = self.model.encoder(obs)
        
        # Ažuriraj std prema rasporedu
        self.std = linear_schedule(self.cfg.std_schedule, step)
        
        horizon = self.cfg.horizon
        mu = self.plan_mu.clone()
        sigma = torch.full_like(mu, 2.0)
        
        total_samples = self.cfg.num_samples + self.cfg.num_pi_samples
        
        for _ in range(self.cfg.iterations):
            z = z0.repeat(total_samples, 1)
            returns = torch.zeros(total_samples, 1, device=self.device)
            # MPPI Rollout
            actions = []
            for t in range(horizon):
                a_rand = mu[t] + sigma[t] * torch.randn(self.cfg.num_samples, self.action_dim, device=self.device)
                a_pi = self.model.policy(z[self.cfg.num_samples:])
                a_pi = a_pi + self.std * torch.randn_like(a_pi)
                a = torch.cat([a_rand, a_pi], dim=0).clamp(-1.0, 1.0)
                actions.append(a)
                
                z_a = torch.cat([z, a], dim=-1)
                z, reward = self.model.dynamics(z_a), self.model.reward(z_a)
                returns += (self.cfg.gamma ** t) * reward
            
            # Terminal Value
            a_term = self.model.policy(z)
            z_a_term = torch.cat([z, a_term], dim=-1)
            q1, q2 = self.model.q_net1(z_a_term), self.model.q_net2(z_a_term)
            returns += (self.cfg.gamma ** horizon) * torch.min(q1, q2)
            
            # Selection of the elites
            actions_stack = torch.stack(actions, dim=1)
            elite_returns, elite_idxs = torch.topk(returns.squeeze(1), self.cfg.elite_size)
            elite_actions = actions_stack[elite_idxs]
            
            # Weighted average
            max_ret = elite_returns[0]
            score = torch.exp(self.cfg.temperature * (elite_returns - max_ret))
            weights = (score / score.sum()).unsqueeze(1).unsqueeze(2)
            
            _mu = torch.sum(weights * elite_actions, dim=0)
            _var = torch.sum(weights * (elite_actions - _mu.unsqueeze(0))**2, dim=0)
            _std = torch.sqrt(_var).clamp(min=self.cfg.min_std)
            
            # Update with momentum
            mu = self.cfg.momentum * mu + (1 - self.cfg.momentum) * _mu
            sigma = _std

        self._prev_mean = mu
        self.plan_mu[:-1] = mu[1:].clone()
        self.plan_mu[-1] = torch.zeros(self.action_dim)
        
        action = mu[0]
        if not eval_mode:
            action += sigma[0] * torch.randn(self.action_dim, device=self.device)
        return action.cpu().numpy()

    def update(self, buffer, step):
        """
        Main learning funtion.
        """
        obs, action, reward = buffer.sample(self.cfg.batch_size, self.cfg.horizon)
        obs = torch.tensor(obs, device=self.device)
        action = torch.tensor(action, device=self.device)
        reward = torch.tensor(reward, device=self.device)

        # Encode the starting latent states
        z = self.model.encoder(obs[:, 0])
        with torch.no_grad():
            z_target_all = self.model_target.encoder(obs)

        zs = [z.detach()]

        consistency_loss = 0.0
        reward_loss = 0.0
        value_loss = 0.0

        for t in range(self.cfg.horizon):
            #Predictions
            z_a = torch.cat([z, action[:, t]], dim=-1)
            z_next = self.model.dynamics(z_a)
            r_pred = self.model.reward(z_a)
            q1, q2 = self.model.q_net1(z_a), self.model.q_net2(z_a)

            #Target
            with torch.no_grad():
                next_z = z_target_all[:, t+1]
                next_a = self.model.policy(next_z)
                z_a_next = torch.cat([next_z, next_a], dim=-1)
                qt1 = self.model_target.q_net1(z_a_next)
                qt2 = self.model_target.q_net2(z_a_next)
                q_target = reward[:, t] + self.cfg.gamma * torch.min(qt1, qt2)

            # Individual Clamping
            l_r = torch.mean((r_pred - reward[:, t])**2).clamp(max=1e4)
            l_q = (torch.mean((q1 - q_target)**2) + torch.mean((q2 - q_target)**2)).clamp(max=1e4)
            l_c = torch.mean((z_next - next_z)**2).clamp(max=1e4)

            rho = self.cfg.rho ** t
            consistency_loss += rho * l_c
            reward_loss += rho * l_r
            value_loss += rho * l_q

            z = z_next
            zs.append(z.detach())

        total_loss = (self.cfg.consistency_coef * consistency_loss +
                      self.cfg.reward_coef * reward_loss +
                      self.cfg.value_coef * value_loss)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        # Policy loss
        # Policy (separated optimizator)
        self.pi_optimizer.zero_grad(set_to_none=True)
        pi_loss = 0.0
        for i, z_i in enumerate(zs[:-1]):
            a_i = self.model.policy(z_i)
            # Ispravka: pozivamo svaku Q mrežu posebno
            q1_i = self.model.q_net1(torch.cat([z_i, a_i], dim=-1))
            q2_i = self.model.q_net2(torch.cat([z_i, a_i], dim=-1))
            q_i = torch.min(q1_i, q2_i)
            pi_loss += -q_i.mean() * (self.cfg.rho ** i)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.cfg.grad_clip_norm)
        self.pi_optimizer.step()

        # EMA of target network
        if step % self.cfg.update_freq == 0:
            ema(self.model, self.model_target, self.cfg.tau)

        return {
            'loss/total': total_loss.item(),
            'loss/consistency': consistency_loss.item(),
            'loss/reward': reward_loss.item(),
            'loss/value': value_loss.item(),
            'loss/pi': pi_loss.item(),
            'std': self.std
        }