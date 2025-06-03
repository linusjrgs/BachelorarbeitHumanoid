import gymnasium as gym
from env import HumanoidCustomEnv
from gymnasium.wrappers import RecordEpisodeStatistics

def make_sb3_env(render_mode=None):
    """
    Factory-Funktion für SB3-kompatible Umgebung mit optionalem render_mode.
    """
    def _init():
        class Wrapper(gym.Env):
            def __init__(self):
                self.env = HumanoidCustomEnv(render_mode=render_mode)
                self.observation_space = self.env.observation_space
                self.action_space = self.env.action_space
                self.metadata = self.env.metadata

            def reset(self, **kwargs):
                obs, _ = self.env.reset(**kwargs)
                return obs, {}

            def step(self, action):
                action = action.clip(-1, 1)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                return obs, reward, done, False, info

            def render(self):
                return self.env.render()

            def close(self):
                self.env.close()

        return RecordEpisodeStatistics(Wrapper())  # Optional: Statistik-Wrapper

    return _init
