import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_env_wrapper import make_sb3_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class HumanoidTrainer:
    def __init__(self, total_timesteps=100000, log_dir="sb3_logs"):
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.video_dir = "video_sb3"
        self.plot_dir = "plot_sb3"
        self.tensorboard_subdir = "tensorboard"
        self.tensorboard_run = "PPO_1"  # default bei Stable-Baselines3

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def train(self):
        env = DummyVecEnv([lambda: Monitor(make_sb3_env()())])
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            tensorboard_log=os.path.join(self.log_dir, self.tensorboard_subdir),
        )

        eval_env = DummyVecEnv([lambda: Monitor(make_sb3_env()())])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.log_dir, "best_model"),
            log_path=self.log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        model.learn(total_timesteps=self.total_timesteps, callback=eval_callback)
        model_path = os.path.join(self.log_dir, "final_model")
        model.save(model_path)
        print(f"\n✅ Training abgeschlossen. Modell gespeichert unter: {model_path}\n")

        self.model = model  # für weitere Nutzung

    def record_video(self, max_steps=1000):
        env = make_sb3_env(render_mode='rgb_array')()
        obs, _ = env.reset()
        frames = []

        for _ in range(max_steps):
            frame = env.render()
            frames.append(frame)

            obs_input = np.expand_dims(obs, axis=0)
            action, _ = self.model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            if done:
                break

        env.close()
        video_path = os.path.join(self.video_dir, "humanoid_run_SB3.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"🎥 Video gespeichert unter: {video_path}")

    def plot_monitor_metrics(self):
        monitor_file = os.path.join(self.log_dir, "monitor.csv")
        if not os.path.exists(monitor_file):
            print("⚠️ Keine Monitor-Datei gefunden – keine Plots erstellt.")
            return

        with open(monitor_file, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if not line.startswith('#')]

        data = np.genfromtxt(lines, delimiter=',', names=True)
        timesteps = np.cumsum(data['l'])
        rewards = data['r']

        plt.figure()
        plt.plot(timesteps, rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title("Training Rewards über Zeit")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "training_rewards.png"))
        plt.savefig(os.path.join(self.plot_dir, "training_rewards.pdf"))
        plt.close()
        print("📊 Trainingsplot gespeichert.")

    def plot_tensorboard_metrics(self):
        log_path = os.path.join(self.log_dir, self.tensorboard_subdir, self.tensorboard_run)
        if not os.path.exists(log_path):
            print(f"⚠️ Kein TensorBoard-Log gefunden unter {log_path}")
            return

        event_acc = EventAccumulator(log_path)
        event_acc.Reload()

        metrics = [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/loss",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/entropy_loss",
            "train/explained_variance"
        ]

        available = event_acc.Tags().get("scalars", [])
        for metric in metrics:
            if metric not in available:
                print(f"⚠️ Metrik nicht gefunden: {metric}")
                continue

            events = event_acc.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            plt.figure()
            plt.plot(steps, values, label=metric)
            plt.xlabel("Timesteps")
            plt.ylabel(metric.split("/")[-1])
            plt.title(metric)
            plt.grid(True)
            plt.tight_layout()

            safe_name = metric.replace("/", "_")
            plt.savefig(os.path.join(self.plot_dir, f"{safe_name}.png"))
            plt.savefig(os.path.join(self.plot_dir, f"{safe_name}.pdf"))
            plt.close()
            print(f"📈 {metric} gespeichert.")


    def run_all(self):
        self.train()
        self.record_video()
        self.plot_monitor_metrics()
        self.plot_tensorboard_metrics()


if __name__ == "__main__":
    trainer = HumanoidTrainer(total_timesteps=100000)
    trainer.run_all()
