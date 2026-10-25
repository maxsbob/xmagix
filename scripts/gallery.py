"""README figures: one scripted-expert episode per embodiment (push) and per task (gripper).

    python scripts/gallery.py      # -> docs/embodiments.png, docs/tasks.png
"""

from pathlib import Path

import jax
import numpy as np

import xmagix
from xmagix.render import episode_grid


def episode(task, emb, seed=1):
    env = xmagix.make(task, emb, 2, auto_reset=False)
    s = env.reset(jax.random.PRNGKey(seed))

    def body(s, k):
        s2, _, info = env.step(s, env.expert(s), k)
        return s2, (s.q, env.effector(s.q), info["success"])

    _, (qs, effs, succ) = jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(seed + 1), env.horizon))
    hit = np.flatnonzero(np.asarray(succ[:, 0]))
    end = env.horizon if isinstance(env.task, xmagix.tasks.Orbit) or not len(hit) else hit[0] + 5
    return env, np.asarray(qs[:end, 0]), np.asarray(effs[:end, 0])


def main():
    out = Path("docs")
    out.mkdir(exist_ok=True)
    items = []
    for e in xmagix.EMBODIMENTS:
        env, qs, eff = episode("push", e)
        items.append((env, qs, eff, f"{e} ({env.act_dim}-d action, {env.obs_dim}-d obs)"))
    episode_grid(items, out / "embodiments.png", cols=4)
    items = []
    for t in xmagix.TASKS:
        env, qs, eff = episode(t, "gripper")
        items.append((env, qs, eff, t))
    episode_grid(items, out / "tasks.png", cols=4)


if __name__ == "__main__":
    main()
