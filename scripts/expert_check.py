"""Scripted-expert success for every task x embodiment (the benchmark's validity check).

    python scripts/expert_check.py [--tasks push sweep] [--embodiments point gripper] [--n 256]
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import xmagix


def evaluate(env, key, noise=0.0):
    """One episode per world from reset; success = solved at any step; score at the end."""
    state = env.reset(key)
    T = env.horizon

    def body(s, k):
        k1, k2 = jax.random.split(k)
        s2, _, info = env.step(s, env.expert(s, k1, noise), k2)
        return s2, (info["success"], info["score"])

    _, (succ, score) = jax.lax.scan(body, state, jax.random.split(key, T))
    return float(jnp.mean(jnp.any(succ, 0))), float(jnp.mean(score[-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(xmagix.TASKS))
    p.add_argument("--embodiments", nargs="+", default=list(xmagix.EMBODIMENTS))
    p.add_argument("--variant", default="default")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    res = {}
    for t in a.tasks:
        for e in a.embodiments:
            env = xmagix.make(t, e, a.n, variant=a.variant, auto_reset=False)
            t0 = time.time()
            succ, score = evaluate(env, jax.random.PRNGKey(0), a.noise)
            res[f"{t}/{e}"] = succ
            print(f"{t:12s} {e:13s} success {succ:.2f}  final score {score:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
