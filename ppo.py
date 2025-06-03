# =========================
# ppo.py – PPOAgent with entropy/KL fixes (GPU‑ready)
# =========================
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from mlp import MLP, MAX_GRAD_NORM
from utils import RunningNorm, explained_variance, safe_atanh

# ──────────────────────────────────────────────────────────────────────────
# Hyper‑parameters
# ──────────────────────────────────────────────────────────────────────────
CLIP_EPS        = 0.2
GAMMA, LAMBDA   = 0.99, 0.95
PI_LR0, VF_LR0  = 3e-4, 3e-4
LR_MAX_PI       = 1e-3
MINI_BATCH      = 256
EPOCHS_MB       = 8
VALUE_COEF      = 1.0
ENT_COEF0       = 1e-2
TARGET_KL       = 0.03
CLIP_VF         = 0.1
NOISE_EPS       = 1e-8
LOG_STD_MIN     = -2.0   # clamp‑Grenzen für log_std
LOG_STD_MAX     = 0.5
ENT_DECAY_RATE  = 0.9995  # exponentielles Decay pro Update
ENT_COEF_MIN    = 1e-4

# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────

def orthogonal_init_(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


# ──────────────────────────────────────────────────────────────────────────
# PPOAgent
# ──────────────────────────────────────────────────────────────────────────
class PPOAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        self.pi = MLP(obs_dim, [256, 256], act_dim)
        self.vf = MLP(obs_dim, [256, 256], 1)

        self.pi.apply(orthogonal_init_)
        self.vf.apply(orthogonal_init_)

        # final layer of policy smaller init (as in OpenAI baselines)
        for module in self.pi.modules():
            if isinstance(module, nn.Linear) and module.out_features == act_dim:
                nn.init.orthogonal_(module.weight, gain=0.01)

        self.pi.to(device)
        self.vf.to(device)

        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5, device=device))
        self.opt_pi = Adam(list(self.pi.parameters()) + [self.log_std], lr=PI_LR0)
        self.opt_vf = Adam(self.vf.parameters(), lr=VF_LR0)

        self.obs_rms = RunningNorm(obs_dim, device=device)

        self.pi_lr = PI_LR0
        self._kl_low_ctr = 0
        self.ent_coef = ENT_COEF0
        self.training_step = 0

    # ------------- helper -------------
    @staticmethod
    def _tanh_correction(a_tanh: torch.Tensor) -> torch.Tensor:
        """Jacobian‑Determinant‑Term für tanh‑Squashing"""
        return torch.log(1.0 - a_tanh.pow(2) + 1e-6)

    # ------------- acting -------------
    def act(self, obs: torch.Tensor | np.ndarray, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        obs_n = self.obs_rms.norm(obs)

        mu = self.pi(obs_n)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        dist = Normal(mu, std)
        a = mu if deterministic else dist.rsample()

        a_tanh = torch.tanh(a)
        logp = dist.log_prob(a).sum(-1) - self._tanh_correction(a_tanh).sum(-1)
        return a_tanh, logp

    # ------------- training -------------
    def update(self, buf: Dict[str, List[torch.Tensor]]) -> Tuple[float, float, float, float]:
        device = self.device

        # --- Buffer in Tensor‑Form bringen ---
        obs   = torch.stack(buf["obs"]).to(device)
        acts  = torch.stack(buf["act"]).to(device).detach()
        old_lp = torch.stack(buf["logp"]).to(device).detach()
        rews  = torch.stack(buf["rew"]).to(device)
        dones = torch.stack(buf["done"]).to(device)
        vals  = torch.stack(buf["val"]).to(device)
        last_val = buf["last_val"].to(device)

        N = last_val.shape[0]            # parallel envs
        T = len(rews) // N              # rollout‑länge

        # reshape zu (T,N,...)
        obs = obs.view(T, N, -1)
        acts = acts.view(T, N, -1)
        old_lp = old_lp.view(T, N)
        rews = rews.view(T, N)
        dones = dones.view(T, N)
        vals = vals.view(T, N)
        vals = torch.cat([vals, last_val.unsqueeze(0)], dim=0)  # (T+1,N)

        # Normalisierung updaten & anwenden
        self.obs_rms.update(obs.reshape(-1, obs.shape[-1]))
        obs_n = self.obs_rms.norm(obs.reshape(-1, obs.shape[-1])).view(T, N, -1)

        # --- GAE-Lambda Advantage ---
        adv = torch.zeros_like(rews)
        last_gae_lam = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rews[t] + GAMMA * vals[t + 1] * mask - vals[t]
            last_gae_lam = delta + GAMMA * LAMBDA * mask * last_gae_lam
            adv[t] = last_gae_lam
        ret = adv + vals[:-1]

        # flatten
        obs_flat = obs_n.reshape(T * N, -1)
        acts_flat = acts.reshape(T * N, -1)
        old_lp_flat = old_lp.reshape(T * N)
        adv_flat = adv.reshape(T * N)
        ret_flat = ret.reshape(T * N)
        v_old_flat = vals[:-1].reshape(T * N)  # value‑prediction zum Zeitpunkt t

        # normalize advantage
        adv_flat = ((adv_flat - adv_flat.mean()) / (adv_flat.std() + NOISE_EPS)).detach()
        ret_flat = ret_flat.detach()

        # --- Entropy‑Decay ---
        self.training_step += 1
        self.ent_coef = max(ENT_COEF_MIN, self.ent_coef * ENT_DECAY_RATE)

        # update LR abhängig von Target‑KL (wird nachher neu gesetzt)
        for g in self.opt_pi.param_groups:
            g["lr"] = self.pi_lr

        idxs = torch.arange(T * N, device=device)
        kl_vals: List[float] = []
        ent_sum, val_loss_sum, mb_counter = 0.0, 0.0, 0

        # --- minibatch‑SGD ---
        for _ in range(EPOCHS_MB):
            shuffled = idxs[torch.randperm(T * N)]
            for start in range(0, T * N, MINI_BATCH):
                mb = shuffled[start:start + MINI_BATCH]
                o_mb   = obs_flat[mb]
                a_mb   = acts_flat[mb]
                lp_old = old_lp_flat[mb]
                adv_mb = adv_flat[mb]
                ret_mb = ret_flat[mb]
                v_old_mb = v_old_flat[mb]

                mu = self.pi(o_mb)
                std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
                dist = Normal(mu, std)

                unsquashed = safe_atanh(a_mb)
                logp = dist.log_prob(unsquashed).sum(-1) - self._tanh_correction(a_mb).sum(-1)

                ratio = torch.exp(logp - lp_old)
                clip_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                loss_pi = -(torch.min(ratio * adv_mb, clip_ratio * adv_mb)).mean()

                entropy = dist.entropy().sum(-1).mean()
                loss = loss_pi - self.ent_coef * entropy

                # --- Policy‑Update ---
                self.opt_pi.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.pi.parameters()) + [self.log_std], MAX_GRAD_NORM)
                self.opt_pi.step()

                # --- Value‑Update mit Clip‑Trick ---
                v_pred = self.vf(o_mb).squeeze(-1)
                v_clip = v_old_mb + (v_pred - v_old_mb).clamp(-CLIP_VF, CLIP_VF)
                loss_v = VALUE_COEF * torch.max((v_pred - ret_mb).pow(2), (v_clip - ret_mb).pow(2)).mean()

                self.opt_vf.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.vf.parameters(), MAX_GRAD_NORM)
                self.opt_vf.step()

                # --- Logging ---
                kl_vals.append((lp_old - logp).abs().mean().item())
                ent_sum += entropy.item()
                val_loss_sum += loss_v.item()
                mb_counter += 1

        kl_mean = float(np.mean(kl_vals)) if kl_vals else 0.0
        entropy_mean = ent_sum / max(mb_counter, 1)
        value_loss_mean = val_loss_sum / max(mb_counter, 1)

        # --- Adaptive LR (KL‑Targeting) ---
        if kl_mean > TARGET_KL * 1.5:
            self.pi_lr = max(self.pi_lr * 0.5, 1e-5)
            self._kl_low_ctr = 0
        elif kl_mean < TARGET_KL * 0.5:
            self._kl_low_ctr += 1
            if self._kl_low_ctr >= 4:
                self.pi_lr = min(self.pi_lr * 1.1, LR_MAX_PI)
                self._kl_low_ctr = 0
        else:
            self._kl_low_ctr = 0

        # --- Explained Variance berechnen ---
        with torch.no_grad():
            v_est = self.vf(obs_flat).squeeze(-1)
        ev = explained_variance(v_est.cpu(), ret_flat.cpu())

        return kl_mean, entropy_mean, ev, value_loss_mean

    # (für externe Nutzung beibehalten)
    explained_variance = staticmethod(explained_variance)
