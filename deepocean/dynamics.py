"""Shared point-mass dynamics for the Deep Ocean arena.

Everything lives inside a unit circle centred at the origin. Agents are simple
damped point masses driven by a 2-D acceleration command; the shark is a
heuristic pursuer. Keeping the physics here (rather than inside the env) lets
both the training environment and the multi-fish demo reuse the exact same
motion, so what you train is what you see.
"""

from __future__ import annotations

import numpy as np

ARENA_RADIUS = 1.0
DT = 0.05

FISH_MAX_SPEED = 1.30
FISH_ACCEL = 6.0
FISH_DAMPING = 1.2

SHARK_MAX_SPEED = 1.12       # a touch slower than the fish, so evasion is possible
SHARK_TURN = 3.2             # steering responsiveness (higher = tighter turns)
CATCH_RADIUS = 0.07          # shark "eats" a fish within this distance


def step_point(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Advance a damped point mass by one timestep and keep it inside the arena."""
    accel = np.clip(accel, -1.0, 1.0)
    vel = vel + accel * FISH_ACCEL * DT
    vel = vel - FISH_DAMPING * vel * DT
    speed = float(np.linalg.norm(vel))
    if speed > FISH_MAX_SPEED:
        vel = vel / speed * FISH_MAX_SPEED
    pos = pos + vel * DT

    # Reflect softly off the circular wall (lose the outward velocity component).
    r = float(np.linalg.norm(pos))
    if r > ARENA_RADIUS:
        n = pos / r
        pos = n * ARENA_RADIUS
        vel = vel - np.dot(vel, n) * n
    return pos, vel


def step_shark(pos: np.ndarray, vel: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Advance the pursuing shark one timestep toward ``target``."""
    to_target = target - pos
    dist = float(np.linalg.norm(to_target))
    if dist > 1e-6:
        desired = to_target / dist * SHARK_MAX_SPEED
    else:
        desired = np.zeros(2)
    vel = vel + (desired - vel) * SHARK_TURN * DT
    speed = float(np.linalg.norm(vel))
    if speed > SHARK_MAX_SPEED:
        vel = vel / speed * SHARK_MAX_SPEED
    pos = pos + vel * DT

    r = float(np.linalg.norm(pos))
    if r > ARENA_RADIUS:
        pos = pos / r * ARENA_RADIUS
    return pos, vel


def random_point_in_arena(rng: np.random.Generator, margin: float = 0.9) -> np.ndarray:
    """Uniform-ish random point well inside the arena."""
    ang = rng.uniform(0, 2 * np.pi)
    rad = ARENA_RADIUS * margin * np.sqrt(rng.uniform(0, 1))
    return np.array([rad * np.cos(ang), rad * np.sin(ang)])
