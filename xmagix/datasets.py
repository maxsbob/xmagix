"""Reward-free play data and expert demonstrations, generated on the GPU in one jitted scan.

play    the scripted planner chases random sub-goals (push a random block to a random place, or
        move the effector to a random point), re-sampled every 25-100 steps, with Gaussian
        action noise and occasional random actions; it knows nothing about the evaluation task.
expert  the task's scripted expert, noise-free; demonstrations keep only successful episodes.

Files use OGBench's keys (observations, actions, terminals, qpos, qvel), so OGBench loaders work.
"""

import jax
import jax.numpy as jnp
import numpy as np

from xmagix.tasks import push_plan


def _play_plan(env, s, ps, key):
    """ps = (goal xy, block index or -1, steps left). Returns (action, new ps)."""
    goal, blk, left = ps
    n = s.q.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    renew = left <= 0
    new_goal = jax.random.uniform(k1, (n, 2), minval=-0.8, maxval=0.8)
    nb = max(env.n_blocks, 1)
    new_blk = jnp.where(jax.random.uniform(k2, (n,)) < (0.7 if env.n_blocks else 0.0),
                        jax.random.randint(k3, (n,), 0, nb), -1)
    goal = jnp.where(renew[:, None], new_goal, goal)
    blk = jnp.where(renew, new_blk, blk)
    left = jnp.where(renew, jax.random.randint(k4, (n,), 25, 100), left - 1)
    eff = env.effector(s.q)
    if env.n_blocks:
        b = jnp.take_along_axis(env.blocks(s.q)[:, :, :2], jnp.maximum(blk, 0)[:, None, None], 1)[:, 0]
        tgt_push, head_push, _ = push_plan(b, goal, eff, env.emb.contact)
    else:
        tgt_push, head_push = goal, jnp.zeros(n)
    d = goal - eff
    use_push = blk >= 0
    target = jnp.where(use_push[:, None], tgt_push, goal)
    heading = jnp.where(use_push, head_push, jnp.arctan2(d[:, 1], d[:, 0]))
    return env.emb.track(s.q, s.v, target, heading), (goal, blk, left)


def collect(env, key, steps, policy="play", noise=0.2, p_random=0.1):
    """Run `env.num_envs` worlds for `steps` control steps with auto-reset. Returns flat arrays
    ordered world by world (episodes contiguous)."""
    n = env.num_envs
    k0, key = jax.random.split(key)
    s0 = env.reset(k0)
    ps0 = (jnp.zeros((n, 2)), jnp.full((n,), -1), jnp.zeros((n,), jnp.int32))

    def body(carry, k):
        s, ps = carry
        k1, k2, k3, k4 = jax.random.split(k, 4)
        if policy == "play":
            a, ps = _play_plan(env, s, ps, k1)
        else:
            a = env.expert(s)
        if noise > 0:
            a = a + noise * jax.random.normal(k2, a.shape)
        rand = jax.random.uniform(k3, (n, 1)) < p_random
        a = jnp.clip(jnp.where(rand, jax.random.uniform(k4, a.shape, minval=-1, maxval=1), a), -1, 1)
        obs = env.observe(s)
        s2, _, info = env.step(s, a, jax.random.fold_in(k, 1))
        rec = dict(observations=obs, actions=a, terminals=info["done"], qpos=s.q, qvel=s.v,
                   success=info["success"])
        return (s2, ps), rec

    _, out = jax.lax.scan(body, (s0, ps0), jax.random.split(key, steps))
    out = jax.device_get(out)
    flat = {k: np.swapaxes(v, 0, 1).reshape(n * steps, *v.shape[2:]) for k, v in out.items()}
    flat["terminals"][steps - 1::steps] = True  # each world's last step closes its last episode
    return flat


def demonstrations(env, key, n_demos=5, max_tries=8):
    """Observation-only expert episodes that solve the task. Goal tasks end two frames after the
    first success (terminated); cyclic tasks (Orbit) run the whole horizon (truncated).
    Returns (observation arrays, qpos arrays, truncated)."""
    from xmagix.tasks import Orbit

    T = env.horizon
    truncated = isinstance(env.task, Orbit)
    obs_out, q_out = [], []
    for attempt in range(max_tries):
        data = collect(env, jax.random.fold_in(key, attempt), T, policy="expert", noise=0.0, p_random=0.0)
        obs = data["observations"].reshape(env.num_envs, T, -1)
        qs = data["qpos"].reshape(env.num_envs, T, -1)
        succ = data["success"].reshape(env.num_envs, T)
        for w in range(env.num_envs):
            hit = np.flatnonzero(succ[w])
            if len(hit):
                end = T if truncated else min(T, hit[0] + 2)
                obs_out.append(obs[w, :end]), q_out.append(qs[w, :end])
            if len(obs_out) == n_demos:
                return obs_out, q_out, truncated
    raise RuntimeError(f"only {len(obs_out)} successful expert episodes for {env.name}")
