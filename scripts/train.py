"""Train a PPO fish to evade the shark.

    python scripts/train.py --timesteps 800000 --n-envs 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import deepocean  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=800_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    (ROOT / "models").mkdir(exist_ok=True)
    train_env = make_vec_env("DeepOcean-v0", n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env("DeepOcean-v0", n_envs=1, seed=args.seed + 1000)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(ROOT / "models" / "best"),
        log_path=str(ROOT / "logs"), eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=15, deterministic=True,
    )
    model = PPO(
        "MlpPolicy", train_env, seed=args.seed,
        n_steps=2048, batch_size=256, n_epochs=10, gamma=0.995,
        gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4,
        policy_kwargs={"net_arch": [128, 128]}, verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(str(ROOT / "models" / "ppo_deepocean"))
    print("saved:", ROOT / "models" / "ppo_deepocean.zip")


if __name__ == "__main__":
    main()
