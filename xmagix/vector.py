"""Gymnasium vector-env view of the functional JAX environment (numpy in, numpy out).

    venv = xmagix.vector.make_vec("push", "gripper", num_envs=1000)
    obs, info = venv.reset(seed=0)
    obs, reward, terminated, truncated, info = venv.step(actions)   # reward = success (0/1)
"""

import gymnasium
import jax
import numpy as np

from xmagix.env import make


class XmagixVectorEnv(gymnasium.vector.VectorEnv):
    def __init__(self, task, embodiment, num_envs, variant="default", backend="warp", **kw):
        self.env = make(task, embodiment, num_envs, variant=variant, backend=backend, **kw)
        self.num_envs = num_envs
        hi = np.ones(self.env.act_dim, np.float32)
        self.single_action_space = gymnasium.spaces.Box(-hi, hi, dtype=np.float32)
        if self.env.obs_type == "pixels":
            self.single_observation_space = gymnasium.spaces.Box(0, 255, self.env.obs_shape, np.uint8)
        else:
            self.single_observation_space = gymnasium.spaces.Box(-np.inf, np.inf, self.env.obs_shape, np.float32)
        self.action_space = gymnasium.vector.utils.batch_space(self.single_action_space, num_envs)
        self.observation_space = gymnasium.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}
        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)
        self._observe = jax.jit(self.env.observe)
        self._state, self._key = None, None

    def reset(self, *, seed=None, options=None):
        self._key = jax.random.PRNGKey(0 if seed is None else int(np.asarray(seed).ravel()[0]))
        self._key, k = jax.random.split(self._key)
        self._state = self._reset(k)
        return np.asarray(self._observe(self._state)), {}

    def step(self, actions):
        self._key, k = jax.random.split(self._key)
        self._state, obs, info = self._step(self._state, np.asarray(actions, np.float32), k)
        succ = np.asarray(info["success"])
        done = np.asarray(info["done"])
        out = dict(success=succ, score=np.asarray(info["score"]), final_obs=np.asarray(info["obs_final"]))
        return np.asarray(obs), succ.astype(np.float32), np.zeros_like(done), done, out


def make_vec(task, embodiment, num_envs=64, **kw):
    return XmagixVectorEnv(task, embodiment, num_envs, **kw)
