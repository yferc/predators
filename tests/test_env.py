"""Behaviour tests for the Deep Ocean environment."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from deepocean import DeepOceanEnv
from deepocean import dynamics as dyn


def test_passes_gym_api_checker():
    check_env(DeepOceanEnv().unwrapped, skip_render_check=True)


def test_spaces():
    env = DeepOceanEnv()
    assert env.observation_space.shape == (9,)
    assert env.action_space.shape == (2,)


def test_reset_deterministic():
    env = DeepOceanEnv()
    o1, _ = env.reset(seed=7)
    o2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(o1, o2)


def test_obs_within_bounds():
    env = DeepOceanEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_passive_fish_gets_caught():
    """A fish that never moves must eventually be caught (episode terminates)."""
    env = DeepOceanEnv()
    env.reset(seed=1)
    caught = False
    for _ in range(600):
        _, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
        if term:
            caught = True
            break
        if trunc:
            break
    assert caught


def test_fleeing_survives_longer_than_passive():
    """A simple flee heuristic should outlast doing nothing — the task is learnable."""
    def survive(policy, seed):
        env = DeepOceanEnv()
        obs, _ = env.reset(seed=seed)
        info = {"survived_steps": 0}
        for _ in range(600):
            obs, _, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                break
        return info["survived_steps"]

    passive = survive(lambda o: np.zeros(2, np.float32), 3)
    flee = survive(lambda o: np.clip(-o[:2] * 8, -1, 1).astype(np.float32), 3)
    assert flee > passive


def test_shark_slower_than_fish():
    """Evasion is only possible if the shark can't simply outrun the fish."""
    assert dyn.SHARK_MAX_SPEED < dyn.FISH_MAX_SPEED
