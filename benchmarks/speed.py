"""Throughput of the batched environment (expert policy in the loop, one jitted lax.scan).

    python benchmarks/speed.py --worlds 1024 4096 16384 --embodiments point gripper arm3
Reports environment steps/s (one step = 5 physics steps) and, for reference, CPU MuJoCo
single-thread steps/s on the same model.
"""

import argparse
import json
import time

import jax
import mujoco
import numpy as np

import xmagix


def gpu_rate(task, emb, n, steps=200):
    env = xmagix.make(task, emb, n)

    def run(key):
        s = env.reset(key)

        def body(s, k):
            s2, _, _ = env.step(s, env.expert(s), k)
            return s2, None

        return jax.lax.scan(body, s, jax.random.split(key, steps))[0].q

    f = jax.jit(run)
    jax.block_until_ready(f(jax.random.PRNGKey(0)))
    t = time.time()
    jax.block_until_ready(f(jax.random.PRNGKey(1)))
    return n * steps / (time.time() - t)


def cpu_rate(task, emb, steps=2000):
    env = xmagix.make(task, emb, 1, backend="cpu")
    m = env.mjm
    d = mujoco.MjData(m)
    t = time.time()
    for _ in range(steps * env.frame_skip):
        mujoco.mj_step(m, d)
    return steps / (time.time() - t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="push")
    p.add_argument("--embodiments", nargs="+", default=["point", "stick_medium", "gripper", "arm3"])
    p.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--json", default=None)
    a = p.parse_args()
    res = {}
    for e in a.embodiments:
        res[e] = dict(cpu_single_thread=cpu_rate(a.task, e))
        for n in a.worlds:
            res[e][str(n)] = gpu_rate(a.task, e, n)
        print(e, {k: f"{v:,.0f}" for k, v in res[e].items()}, flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
