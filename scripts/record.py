"""Record a stylish swarm demo: many fish sharing one trained brain.

Every fish reacts to the shark from its own point of view using the same policy
trained on the single-agent env, and the shark chases the nearest living fish.

    python scripts/record.py --model models/best/best_model.zip --out docs/media/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from deepocean import dynamics as dyn
from deepocean.render import Renderer

ROOT = Path(__file__).resolve().parent.parent


def fish_obs(fp, fv, sp, sv):
    """Egocentric observation for one fish — identical layout to DeepOceanEnv."""
    rel = (sp - fp) / (2 * dyn.ARENA_RADIUS)
    return np.concatenate([
        rel, sv / dyn.SHARK_MAX_SPEED, fv / dyn.FISH_MAX_SPEED,
        fp / dyn.ARENA_RADIUS,
        [np.linalg.norm(sp - fp) / (2 * dyn.ARENA_RADIUS)],
    ]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-fish", type=int, default=18)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=520)
    p.add_argument("--jitter", type=float, default=0.25,
                   help="per-fish action noise to decorrelate the shared brain")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    rng = np.random.default_rng(args.seed)

    fp = np.array([dyn.random_point_in_arena(rng, 0.85) for _ in range(args.n_fish)])
    fv = np.zeros((args.n_fish, 2))
    alive = np.ones(args.n_fish, dtype=bool)
    sp = np.zeros(2)
    sv = np.zeros(2)

    r = Renderer("rgb_array")
    frames = []
    for frame_i in range(args.max_frames):
        obs = np.array([fish_obs(fp[i], fv[i], sp, sv) for i in range(args.n_fish)])
        # Sample the policy stochastically and add a little per-fish noise so the
        # shared brain produces an individual, spread-out school rather than a
        # single stacked trajectory.
        acts, _ = model.predict(obs, deterministic=False)
        acts = np.clip(acts + rng.normal(0, args.jitter, acts.shape), -1.0, 1.0)
        for i in range(args.n_fish):
            if alive[i]:
                fp[i], fv[i] = dyn.step_point(fp[i], fv[i], acts[i])

        # Shark hunts the nearest living fish.
        living = np.where(alive)[0]
        if len(living) == 0:
            break
        target = fp[living[np.argmin(np.linalg.norm(fp[living] - sp, axis=1))]]
        sp, sv = dyn.step_shark(sp, sv, target)
        for i in living:
            if np.linalg.norm(fp[i] - sp) < dyn.CATCH_RADIUS:
                alive[i] = False

        frames.append(r.draw(
            fish=[(fp[i], fv[i], bool(alive[i])) for i in range(args.n_fish)],
            shark=(sp, sv), hud=f"t {frame_i * dyn.DT:4.1f}s",
        ))
    r.close()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(f"{out}.gif", [f[::2, ::2] for f in frames], fps=args.fps, loop=0)
    imageio.mimsave(f"{out}.mp4", frames, fps=args.fps, quality=8)
    print(f"{out}.gif / .mp4  ({len(frames)} frames, {int(alive.sum())}/{args.n_fish} survived)")


if __name__ == "__main__":
    main()
