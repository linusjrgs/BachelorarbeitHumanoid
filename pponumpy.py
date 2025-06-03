import numpy as np
from mlpnumpy import DeepMLP

CLIP_EPS        = 0.2
GAMMA           = 0.99
LAMBDA          = 0.95
PI_LR           = 3e-4
VF_LR           = 3e-4
MINI            = 256
EPOCH_MB        = 8
VALUE_COEF      = 0.5
ENTROPY_COEF0   = 0.05
ENTROPY_COEF_MIN= 0.001
TARGET_KL       = 0.03
CLIP_RANGE_VF   = 0.2


class RunningNorm:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape)
        self.var  = np.ones(shape)
        self.count= eps

    def update(self, x: np.ndarray):
        batch_mean = x.mean(0)
        batch_var  = x.var(0)
        batch_count= x.shape[0]
        delta      = batch_mean - self.mean
        tot        = self.count + batch_count
        new_mean   = self.mean + delta * batch_count / tot
        m_a        = self.var * self.count
        m_b        = batch_var * batch_count
        new_var    = (m_a + m_b + delta ** 2 * self.count * batch_count / tot) / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

    def norm(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class PPOAgent:
    """Fast‑rein NumPy PPO (korrekter PG, KL‑Kontrolle, Entropie)"""

    def __init__(self, obs_dim: int, act_dim: int):
        self.pi = DeepMLP(obs_dim, [256, 256], act_dim)
        self.vf = DeepMLP(obs_dim, [256, 256], 1)
        self.log_std = np.full(act_dim, -0.5)
        self.pi_lr   = PI_LR
        self.vf_lr   = VF_LR
        self.obs_rms = RunningNorm(obs_dim)
        self.ret_rms = RunningNorm(1)

    # ---------- Interaktion ----------
    def _forward_policy(self, obs: np.ndarray):
        mu = self.pi.forward(obs)
        std = np.exp(self.log_std)
        return mu, std

    def act(self, obs: np.ndarray):
        o_n = self.obs_rms.norm(obs)
        mu, std = self._forward_policy(o_n)
        noise   = np.random.randn(*mu.shape)
        a       = mu + noise * std
        logp    = -0.5 * (((a - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi)).sum(-1)
        return a, logp

    def explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Berechnet die erklärte Varianz zwischen Prädiktion und Ziel."""
        var_y = np.var(y_true)
        return 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else np.nan


    # ---------- GAE ----------
    def _gae(self, r, v, d, last_v):
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            mask  = 1.0 - d[t]
            delta = r[t] + GAMMA * v[t + 1] * mask - v[t]
            gae   = delta + GAMMA * LAMBDA * mask * gae
            adv[t]= gae
        ret = adv + v[:-1]
        self.ret_rms.update(ret[:, None])
        return adv, ret

    # ---------- Training ----------
    def update(self, buf: dict, epoch: int):
        entropy_coef = max(ENTROPY_COEF_MIN, ENTROPY_COEF0 * (0.98 ** epoch))

        obs  = np.asarray(buf['obs'])
        act  = np.asarray(buf['act'])
        logp_old = np.asarray(buf['logp'])
        adv  = np.asarray(buf['adv'])
        ret  = np.asarray(buf['ret'])

        # Normalisierung
        self.obs_rms.update(obs)
        obs_n = self.obs_rms.norm(obs)
        ret_n = self.ret_rms.norm(ret[:, None]).squeeze()
        adv   = (adv - adv.mean()) / (adv.std() + 1e-8)

        inds = np.arange(len(obs))
        approx_kl_running = 0.0
        num_mb = 0

        for _ in range(EPOCH_MB):
            np.random.shuffle(inds)
            for s in range(0, len(obs), MINI):
                mb_idx = inds[s:s + MINI]
                mb_o   = obs_n[mb_idx]
                mb_a   = act[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret_n[mb_idx]

                # ------ Forward ------
                mu, std    = self._forward_policy(mb_o)
                var        = std ** 2
                logp       = -0.5 * (((mb_a - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi)).sum(-1)
                ratio      = np.exp(logp - mb_logp_old)

                # ------ Clips ------
                unclipped  = ratio * mb_adv
                clipped    = np.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                loss_pi_sample = -np.minimum(unclipped, clipped)
                loss_pi = loss_pi_sample.mean() - entropy_coef * (self.log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()

                # ------ GRADIENTE mu / log_std ------
                # Indikator: wo wurde geklippt?
                not_clip_mask = ((mb_adv >= 0) & (ratio < 1 + CLIP_EPS)) | ((mb_adv < 0) & (ratio > 1 - CLIP_EPS))
                # shape (mini, act_dim)
                g_logp_mu = (mb_a - mu) / var  # deriv von logp nach mu   (VOR minus‑Zeichen!)
                g_logp_logstd = ((mb_a - mu) ** 2) / var - 1             # deriv logp nach logσ

                #   dL/dmu  = -adv * ratio * g_logp_mu   (falls nicht geklippt)
                grad_mu = -(mb_adv * ratio * not_clip_mask)[:, None] * g_logp_mu / MINI
                #   dL/dlogσ = -adv*ratio*g_logp_logstd - entropy_coef
                grad_logstd = (-(mb_adv * ratio * not_clip_mask)[:, None] * g_logp_logstd) / MINI
                grad_logstd = grad_logstd.sum(axis=0) - entropy_coef  # sum über Batch

                # ------ Backprop ------
                self.pi.backward(grad_mu)
                self.pi.step(self.pi_lr)

                # ----- Wertfunktion (mit VF‑Clip) -----
                v = self.vf.forward(mb_o).squeeze()
                v_clip = np.clip(v, mb_ret - CLIP_RANGE_VF, mb_ret + CLIP_RANGE_VF)
                loss_v1 = (v - mb_ret) ** 2
                loss_v2 = (v_clip - mb_ret) ** 2
                mask_v  = (loss_v1 >= loss_v2).astype(float)
                grad_v  = 2 * VALUE_COEF * ((v - mb_ret) * mask_v + (v_clip - mb_ret) * (1 - mask_v)) / MINI
                self.vf.backward(grad_v[:, None])
                self.vf.step(self.vf_lr)

                # --- log_std Schritt ---
                self.log_std -= self.pi_lr * grad_logstd
                self.log_std = np.clip(self.log_std, -1.5, 0.5)

                approx_kl_running += (mb_logp_old - logp).mean()
                num_mb += 1

                if approx_kl_running / num_mb > TARGET_KL * 2:
                    print(f"⚠️ Frühzeitiger Stopp wegen KL={(approx_kl_running/num_mb):.4f}")
                    return approx_kl_running / num_mb

        return approx_kl_running / max(1, num_mb)