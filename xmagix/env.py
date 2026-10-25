"""Functional batched environment: every method is a pure JAX function of an EnvState, so
rollouts, data collection and evaluation compile into one program.

    env = xmagix.make("push", "gripper", num_envs=4096)
    state = env.reset(jax.random.PRNGKey(0))
    state, obs, info = env.step(state, action)       # info: success, score, done
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from xmagix.embodiments import EMBODIMENTS, Embodiment, rot
from xmagix.scene import build_xml
from xmagix.sim import CpuSim, WarpSim
from xmagix.tasks import TASKS, Task


class EnvState(NamedTuple):
    q: jax.Array  # (N, nq) qpos: embodiment joints first, then 3 per block
    v: jax.Array  # (N, nv)
    ts: jax.Array  # (N, k) task state (gate passed, laps, waypoint index)
    t: jax.Array  # (N,) step within the episode
    eff: jax.Array  # (N, 2) effector position (for path-dependent tasks)
    done: jax.Array  # (N,) the episode just ended (the state has been reset if auto_reset)


@dataclass
class Env:
    task: Task
    emb: Embodiment
    num_envs: int
    variant: str = "default"
    backend: str = "warp"
    frame_skip: int = 5
    auto_reset: bool = True
    obs_type: str = "state"  # state | pixels (allocentric top-down image, uint8, see xmagix.draw)
    pixel_size: int = 64

    def __post_init__(self):
        self.xml = build_xml(self.task, self.emb, self.variant)
        sim = WarpSim if self.backend == "warp" else CpuSim
        self.sim = sim(self.xml, self.num_envs, self.frame_skip)
        self.mjm = self.sim.mjm
        self.n_blocks = len(self.task.blocks)
        self.act_dim = self.emb.act_dim
        self.obs_dim = int(self.state_obs(self._zeros()).shape[-1])  # state features
        self.obs_shape = (self.pixel_size, self.pixel_size, 3) if self.obs_type == "pixels" else (self.obs_dim,)
        self.horizon = int(round(self.task.horizon * self.emb.time_scale))

    @property
    def name(self):
        return f"{self.task.name}-{self.emb.name}" + ("" if self.variant == "default" else f"-{self.variant}")

    # ------------------------------------------------------------------ state helpers
    def _zeros(self):
        n = self.num_envs
        z = jnp.zeros((n, self.mjm.nq), jnp.float32)
        return EnvState(z, jnp.zeros((n, self.mjm.nv), jnp.float32), jnp.zeros((n, self.task.n_task_state)),
                        jnp.zeros(n, jnp.int32), jnp.zeros((n, 2)), jnp.zeros(n, bool))

    def blocks(self, q):
        return q[:, self.emb.nq:].reshape(q.shape[0], self.n_blocks, 3)

    def effector(self, q):
        return self.emb.effector(q)

    def _observe(self, s):
        if self.obs_type == "pixels":
            from xmagix.draw import render

            return render(self, s.q, s.v, size=self.pixel_size)
        return self.state_obs(s)

    def state_obs(self, s):
        """Embodiment proprioception + every block in the embodiment's own frame (position,
        and orientation as cos/sin of 4x the relative yaw: a square looks the same every 90°)."""
        xy, yaw = self.emb.pose(s.q)
        parts = [self.emb.proprio(s.q, s.v)]
        if self.n_blocks:
            b = self.blocks(s.q)
            rel = rot(yaw[:, None], b[:, :, :2] - xy[:, None])
            ang = 4 * (b[:, :, 2] - yaw[:, None])
            parts.append(jnp.concatenate([rel, jnp.cos(ang)[..., None], jnp.sin(ang)[..., None]], -1).reshape(
                s.q.shape[0], -1))
        return jnp.concatenate(parts, -1)

    observe = _observe

    def _initial(self, key):
        k1, k2 = jax.random.split(key)
        n = self.num_envs
        aq = self.emb.init(k1, n, jnp.asarray(self.task.agent_start[0]), self.task.agent_start[1])
        bq = self.task.init_blocks(k2, n).reshape(n, -1)
        q = jnp.concatenate([aq, bq], -1).astype(jnp.float32)
        eff = self.emb.effector(q)
        return EnvState(q, jnp.zeros((n, self.mjm.nv), jnp.float32), jnp.zeros((n, self.task.n_task_state)),
                        jnp.zeros(n, jnp.int32), eff, jnp.zeros(n, bool))

    # ------------------------------------------------------------------ API
    def reset(self, key):
        return self._initial(key)

    def step(self, s, action, key=None):
        """One control step (frame_skip physics steps). With auto_reset, finished worlds are
        re-initialised from `key` (required then) and `info` describes the finished episode."""
        a = jnp.clip(action, -1, 1).astype(jnp.float32)
        ctrl = self.emb.ctrl(s.q, s.v, a).astype(jnp.float32)
        q, v = self.sim(s.q, s.v, ctrl)
        eff = self.emb.effector(q)
        ts = self.task.update(s.ts, self.blocks(q), eff, s.eff)
        t = s.t + 1
        success = self.task.success(self.blocks(q), eff, ts)
        score = self.task.score(self.blocks(q), eff, ts)
        done = t >= self.horizon
        nxt = EnvState(q, v, ts, t, eff, done)
        obs = self._observe(nxt)
        info = dict(success=success, score=score, done=done, obs_final=obs)
        if self.auto_reset:
            fresh = self._initial(key)
            sel = lambda x, y: jnp.where(done.reshape(-1, *[1] * (x.ndim - 1)), y, x)  # noqa: E731
            nxt = jax.tree.map(sel, nxt, fresh)._replace(done=done)
            obs = self._observe(nxt)
        return nxt, obs, info

    def expert(self, s, key=None, noise=0.0):
        """Scripted expert: the task's plan for the effector, tracked by the embodiment."""
        target, heading, _ = self.task.plan(self.blocks(s.q), self.emb.effector(s.q), s.ts, self.emb.contact)
        a = self.emb.track(s.q, s.v, target, heading)
        if noise > 0:
            a = a + noise * jax.random.normal(key, a.shape)
        return jnp.clip(a, -1, 1)


def make(task, embodiment, num_envs=1024, variant="default", backend="warp", **kw):
    """`obs_type="pixels"` (with `pixel_size`) gives top-down images instead of state features."""
    return Env(TASKS[task], EMBODIMENTS[embodiment], num_envs, variant=variant, backend=backend, **kw)


def rollout(env, policy, key, steps, state=None):
    """Run `policy(obs, state, key) -> action` for `steps` steps (auto-reset on). Returns the
    final state and per-step dict(obs, action, success, score, done, q)."""
    k0, key = jax.random.split(key)
    state = env.reset(k0) if state is None else state

    def body(carry, k):
        s = carry
        k1, k2 = jax.random.split(k)
        obs = env.observe(s)
        a = policy(obs, s, k1)
        s2, _, info = env.step(s, a, k2)
        return s2, dict(obs=obs, action=a, q=s.q, success=info["success"], score=info["score"], done=info["done"])

    return jax.lax.scan(body, state, jax.random.split(key, steps))
