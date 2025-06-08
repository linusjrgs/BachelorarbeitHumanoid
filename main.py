"""
Trainings-Skript für den Walking-Only-Humanoid (Proximal Policy Optimization).

Dieses Skript dient als zentraler Einstiegspunkt zum Trainieren eines PPO-Agenten
auf einer angepassten Humanoid-Umgebung, die auf symmetrisches, natürliches
Gangverhalten optimiert wurde.

Merkmale:
- CLI-Argument --tag → eigene TensorBoard-Ordner pro Run (für Vergleichbarkeit)
- STEPS_PER_EPOCH = 2048 → stabileres Update durch mehr Samples pro Policy-Update
- VIDEO_FPS = 60 → flüssige Videos zur qualitativen Evaluation
- Logging in TensorBoard & CSV → quantitative Nachvollziehbarkeit
- Regelmäßiges Speichern von Checkpoints → Wiederaufnahme und Analyse möglich
- Automatische Videoaufzeichnung bei vordefinierten Epochen → Fortschritt visualisieren

Trainingsablauf:
- Trainingsdaten werden parallel in NUM_ENVS Umgebungen gesammelt (AsyncVectorEnv)
- Training erfolgt über EPOCHS Schleifen → 1 PPO-Update pro EPOCH
- Nach jeweils EVAL_INTERVAL Epochen erfolgt eine Evaluation der aktuellen Policy im Einzelmodus
- Policy-Verbesserungen und Lernfortschritte werden zusätzlich in Videos sichtbar gemacht

Ziel:
- Entwicklung eines stabilen, symmetrischen und natürlichen Gangs für den Humanoid-Agenten.
"""

from __future__ import annotations
import argparse, datetime, os
import imageio, matplotlib.pyplot as plt, numpy as np, pandas as pd, torch
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

from env import make_env
from ppo import PPOAgent

# ────────────────────────── CLI ──────────────────────────
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--tag", type=str, default="walk", help="Name‑Prefix für TensorBoard‑Run")
PARSER.add_argument("--seed", type=int, default=None, help="Zufalls-Seed (None = zufällig)")
ARGS = PARSER.parse_args()

# ────────────────────────── HYPER‑PARAMS ─────────────────────────
NUM_ENVS        = 16
EPOCHS          = 20000
STEPS_PER_EPOCH = 2048   # vorher 1024
VIDEO_EPOCHS    = VIDEO_EPOCHS = {50, 100, 180, 260, 350, 420, 430, 499, 600, 700, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 23000, 25000, 27000, 28000, 29000, 29999}
VIDEO_FPS       = 60
SEED = ARGS.seed if ARGS.seed is not None else np.random.randint(0, 10_000)
# ───────────────────── Helper ─────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def evaluate_policy(agent: PPOAgent, n_ep: int = 3, max_steps: int = 1500) -> float:
    env = make_env()
    returns: list[float] = []
    for _ in range(n_ep):
        obs, _ = env.reset(seed=None)
        done, R, steps = False, 0.0, 0
        while not done and steps < max_steps:
            with torch.no_grad():
                act, _ = agent.act(torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device), deterministic=True)
            obs, r, term, trunc, _ = env.step(_to_numpy(act).squeeze())
            R += r
            done = bool(term or trunc)
            steps += 1
        returns.append(R)
    env.close()
    return float(np.mean(returns))


def save_checkpoint(agent: PPOAgent, fname: str) -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    torch.save({
        "pi_state_dict": agent.pi.state_dict(),
        "vf_state_dict": agent.vf.state_dict(),
        "log_std": agent.log_std.detach().cpu(),
    }, fname)

# ─────────────────────── Training ───────────────────────

def train() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    run_dir = f"runs/{ARGS.tag}_seed{SEED}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer  = SummaryWriter(run_dir)
    for d in ("videos", "logs", "plots", "checkpoints"): os.makedirs(d, exist_ok=True)

    vec_env = AsyncVectorEnv([lambda i=i: make_env(seed=i + SEED) for i in range(NUM_ENVS)])
    obs, _  = vec_env.reset(seed=SEED)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    agent   = PPOAgent(obs.shape[-1], vec_env.single_action_space.shape[0], device)

    log_epoch = {k: [] for k in (
        "episode_len", "reward", "val_loss", "policy_std", "log_std", "explained_var",
        "max_reward", "kl_div", "eval_rew", "entropy", "pi_lr")}
    log_step = {"reward": []}

    for ep in range(EPOCHS):
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        obs, _ = vec_env.reset()

        # Neue Variablen für episodisches Messen
        step_count = 0
        ep_lens = []
        ep_len_tracker = np.zeros(NUM_ENVS, dtype=int)

        while step_count < STEPS_PER_EPOCH:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            act_t, logp_t = agent.act(obs_t)
            nxt, r, term, trunc, _ = vec_env.step(_to_numpy(act_t))
            with torch.no_grad():
                v_pred = agent.vf(agent.obs_rms.norm(obs_t)).squeeze(-1)

            buf["obs"].extend(obs_t)
            buf["act"].extend(act_t)
            buf["logp"].extend(logp_t)
            buf["rew"].extend(torch.as_tensor(r, dtype=torch.float32, device=device))
            buf["done"].extend(torch.as_tensor(term | trunc, dtype=torch.float32, device=device))
            buf["val"].extend(v_pred)

            ep_len_tracker += 1
            for i, done in enumerate(term | trunc):
                if done:
                    ep_lens.append(ep_len_tracker[i])
                    ep_len_tracker[i] = 0

            obs = nxt
            step_count += NUM_ENVS

        with torch.no_grad():
            last_v = agent.vf(agent.obs_rms.norm(torch.as_tensor(obs, dtype=torch.float32, device=device))).squeeze(-1)
        buf["last_val"] = last_v

        kl, entropy, expl_var, v_loss = agent.update(buf)

        rewards    = torch.stack(buf["rew"]).cpu().numpy()
        mean_rew   = float(rewards.mean())
        policy_std = float(torch.exp(agent.log_std).mean())
        log_std_m  = float(agent.log_std.mean())
        max_rew    = float(rewards.max())
        eval_rew   = evaluate_policy(agent) if ep % 10 == 0 else float("nan")
        ep_len     = int(np.mean(ep_lens)) if ep_lens else STEPS_PER_EPOCH

        log_epoch["episode_len"].append(ep_len)
        log_epoch["reward"].append(mean_rew)
        log_epoch["val_loss"].append(v_loss)
        log_epoch["policy_std"].append(policy_std)
        log_epoch["log_std"].append(log_std_m)
        log_epoch["explained_var"].append(expl_var)
        log_epoch["max_reward"].append(max_rew)
        log_epoch["kl_div"].append(kl)
        log_epoch["eval_rew"].append(eval_rew)
        log_epoch["entropy"].append(entropy)
        log_epoch["pi_lr"].append(agent.pi_lr)
        log_step["reward"].append(mean_rew)

        print(
            f"Ep {ep:<4} | Len {ep_len:<4} | Rew {mean_rew:>6.2f} | "
            f"KL {kl:>7.5f} | VF {v_loss:>8.4f} | Stdπ {policy_std:>6.4f} | "
            f"log_std {log_std_m:>7.4f} | ExplVar {expl_var:>5.3f} | Ent {entropy:>6.3f}",
            flush=True,
        )

        writer.add_scalar("Reward/Mean", mean_rew, ep)
        writer.add_scalar("Reward/Max", max_rew, ep)
        if not np.isnan(eval_rew): writer.add_scalar("Reward/Eval", eval_rew, ep)
        writer.add_scalar("Episode/Length", ep_len, ep)
        writer.add_scalar("Loss/VF", v_loss, ep)
        writer.add_scalar("Value/ExplVar", expl_var, ep)
        writer.add_scalar("Policy/Std", policy_std, ep)
        writer.add_scalar("Policy/log_std", log_std_m, ep)
        writer.add_scalar("Policy/KL", kl, ep)
        writer.add_scalar("Policy/Entropy", entropy, ep)
        writer.add_scalar("Policy/pi_lr", agent.pi_lr, ep)

        if ep in VIDEO_EPOCHS:
            env_v, frames = make_env(render_mode="rgb_array"), []
            o = env_v.reset(seed=None)[0]
            for _ in range(4500):
                with torch.no_grad():
                    a, _ = agent.act(torch.as_tensor(o[None], dtype=torch.float32, device=device), deterministic=True)
                o, _, t, tr, _ = env_v.step(_to_numpy(a).squeeze())
                frames.append(env_v.render())
                if t or tr:
                    break
            imageio.mimsave(f"videos/run_ep_{ARGS.tag}{ep}.mp4", frames, fps=VIDEO_FPS)
            env_v.close()

        if ep % 10 == 0:
            pd.DataFrame(log_epoch).to_csv("logs/training_log_epoch.csv", index_label="epoch")
        if ep % 50 == 0:
            pd.DataFrame(log_step).to_csv("logs/training_log_step.csv", index=False)
            for k, v in log_epoch.items():
                plt.figure(); plt.plot(v); plt.title(k); plt.savefig(f"plots/{k}.png"); plt.close()
        if ep % 100 == 0:
            save_checkpoint(agent, f"checkpoints/ppo_ep{ep}.pt")

    save_checkpoint(agent, "checkpoints/ppo_final.pt")
    writer.close(); print("Training complete.")


if __name__ == "__main__":
    train() 
