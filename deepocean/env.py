"""DeepOcean-v0 — learn to evade a pursuing shark in a circular arena.

A single fish (the agent) swims inside a unit circle while a heuristic shark
chases it. The fish sees the shark and the wall relative to itself and must
survive as long as possible. The learned policy is deliberately *egocentric*
(everything is relative to the fish), so at demo time the same brain can drive a
whole swarm — each fish reacting to the shark from its own point of view.

Observation (Box, shape=(9,)):
    rel. shark position (2), rel. shark velocity (2), own velocity (2),
    own radial position (2, how close to the wall), shark distance (1)

Action (Box, shape=(2,), [-1, 1]): 2-D acceleration.

Reward: + survive, + keep distance from the shark, - hug the wall,
        - large penalty when caught (episode ends).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import dynamics as dyn


class DeepOceanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, max_steps: int = 600) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        high = np.ones(9, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.fish_pos = np.zeros(2)
        self.fish_vel = np.zeros(2)
        self.shark_pos = np.zeros(2)
        self.shark_vel = np.zeros(2)
        self._steps = 0
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.fish_pos = dyn.random_point_in_arena(self.np_random, 0.6)
        self.fish_vel = np.zeros(2)
        # Shark starts on the far side so the first frames aren't an instant loss.
        self.shark_pos = -self.fish_pos / (np.linalg.norm(self.fish_pos) + 1e-9) * 0.9
        self.shark_vel = np.zeros(2)
        self._steps = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        self.fish_pos, self.fish_vel = dyn.step_point(
            self.fish_pos, self.fish_vel, np.asarray(action, dtype=float))
        self.shark_pos, self.shark_vel = dyn.step_shark(
            self.shark_pos, self.shark_vel, self.fish_pos)
        self._steps += 1

        dist = float(np.linalg.norm(self.shark_pos - self.fish_pos))
        radial = float(np.linalg.norm(self.fish_pos)) / dyn.ARENA_RADIUS

        reward = 0.05                                   # survival
        reward += 0.02 * min(dist, 0.6)                 # keep your distance
        if radial > 0.9:                                # don't cower on the wall
            reward -= 0.03 * (radial - 0.9) / 0.1

        terminated = False
        if dist < dyn.CATCH_RADIUS:
            reward -= 10.0
            terminated = True
        truncated = self._steps >= self.max_steps

        info = {"survived_steps": self._steps, "shark_distance": dist}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .render import Renderer
            self._renderer = Renderer(self.render_mode)
        return self._renderer.draw(
            fish=[(self.fish_pos, self.fish_vel, True)],
            shark=(self.shark_pos, self.shark_vel),
            hud=f"t {self._steps * dyn.DT:4.1f}s",
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _obs(self) -> np.ndarray:
        rel = (self.shark_pos - self.fish_pos) / (2 * dyn.ARENA_RADIUS)
        svel = self.shark_vel / dyn.SHARK_MAX_SPEED
        fvel = self.fish_vel / dyn.FISH_MAX_SPEED
        radial = self.fish_pos / dyn.ARENA_RADIUS
        dist = np.array([np.linalg.norm(self.shark_pos - self.fish_pos) / (2 * dyn.ARENA_RADIUS)])
        return np.concatenate([rel, svel, fvel, radial, dist]).astype(np.float32)
