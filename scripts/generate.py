"""Generate play datasets and expert demonstrations for task x embodiment pairs.

    python scripts/generate.py --tasks push --embodiments point gripper --size 1000000
writes data/<task>-<embodiment>[-variant]/{play.npz, demos.npz}.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import xmagix
from xmagix.datasets import collect, demonstrations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--embodiments", nargs="+", default=None)
    p.add_argument("--variant", default="default")
    p.add_argument("--valid_only", action="store_true", help="only pairs whose expert succeeds >= 90%")
    p.add_argument("--size", type=int, default=1_000_000, help="play transitions per pair")
    p.add_argument("--worlds", type=int, default=1000)
    p.add_argument("--demos", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    pairs = xmagix.valid_pairs() if a.valid_only else [
        (t, e) for t in (a.tasks or xmagix.TASKS) for e in (a.embodiments or xmagix.EMBODIMENTS)]
    pairs = [(t, e) for t, e in pairs if (a.tasks is None or t in a.tasks) and (a.embodiments is None or e in a.embodiments)]
    for t, e in pairs:
        t0 = time.time()
        d = a.out / (f"{t}-{e}" + ("" if a.variant == "default" else f"-{a.variant}"))
        d.mkdir(parents=True, exist_ok=True)
        key = jax.random.PRNGKey(a.seed)
        env = xmagix.make(t, e, a.worlds, variant=a.variant)
        play = collect(env, key, a.size // a.worlds, policy="play")
        np.savez_compressed(d / "play.npz", **{k: v for k, v in play.items() if k != "success"})
        demo_env = xmagix.make(t, e, 64, variant=a.variant)
        obs, qs, truncated = demonstrations(demo_env, jax.random.fold_in(key, 1), a.demos)
        np.savez_compressed(d / "demos.npz", truncated=truncated,
                            **{f"observations_{i}": o for i, o in enumerate(obs)},
                            **{f"qpos_{i}": q for i, q in enumerate(qs)})
        meta = dict(task=t, embodiment=e, variant=a.variant, obs_dim=env.obs_dim, act_dim=env.act_dim,
                    horizon=env.horizon, play_transitions=len(play["actions"]),
                    play_success_rate=float(play["success"].mean()), demo_lengths=[len(o) for o in obs],
                    truncated_demos=truncated)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{t}-{e}: {len(play['actions'])} play transitions, demos {meta['demo_lengths']} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
