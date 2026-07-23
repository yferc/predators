"""Deep Ocean — a top-down shark-evasion RL environment."""

from gymnasium.envs.registration import register

from .env import DeepOceanEnv

__version__ = "0.1.0"
__all__ = ["DeepOceanEnv"]

register(
    id="DeepOcean-v0",
    entry_point="deepocean.env:DeepOceanEnv",
    max_episode_steps=600,
)
