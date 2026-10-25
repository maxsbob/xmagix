"""Batched physics as a JAX function (qpos, qvel, ctrl) -> (qpos, qvel) after `frame_skip`
MuJoCo steps.

WarpSim: MJWarp on the GPU, exposed to JAX with `warp.jax_callable`, so a policy rollout is one
`jax.jit`/`lax.scan` program and never leaves the device. The physics state is carried as JAX
arrays and copied into a persistent `mjw.Data` around each call (microseconds for these models).
CpuSim: the same function with CPU MuJoCo (a reference for tests; slow).
"""

import mujoco
import numpy as np


class WarpSim:
    def __init__(self, xml, nworld, frame_skip=5, nconmax=32, njmax=160):
        import mujoco_warp as mjw
        import warp as wp

        wp.config.quiet = True
        wp.init()
        self.mjm = mujoco.MjModel.from_xml_string(xml)
        self.nworld, self.frame_skip = nworld, frame_skip
        m = mjw.put_model(self.mjm)
        d = mjw.put_data(self.mjm, mujoco.MjData(self.mjm), nworld=nworld, nconmax=nconmax, njmax=njmax)
        self._m, self._d = m, d  # keep alive: captured by the callable below

        def step(qpos_in: wp.array2d(dtype=wp.float32), qvel_in: wp.array2d(dtype=wp.float32),
                 ctrl_in: wp.array2d(dtype=wp.float32),
                 qpos_out: wp.array2d(dtype=wp.float32), qvel_out: wp.array2d(dtype=wp.float32)):
            wp.copy(d.qpos, qpos_in)
            wp.copy(d.qvel, qvel_in)
            wp.copy(d.ctrl, ctrl_in)
            for _ in range(frame_skip):
                mjw.step(m, d)
            wp.copy(qpos_out, d.qpos)
            wp.copy(qvel_out, d.qvel)

        self._step = wp.jax_callable(step, num_outputs=2,
                                     output_dims={"qpos_out": (nworld, self.mjm.nq), "qvel_out": (nworld, self.mjm.nv)},
                                     graph_mode=wp.JaxCallableGraphMode.WARP)

    def __call__(self, q, v, c):
        return self._step(q, v, c)


class CpuSim:
    def __init__(self, xml, nworld, frame_skip=5):
        self.mjm = mujoco.MjModel.from_xml_string(xml)
        self.nworld, self.frame_skip = nworld, frame_skip
        self._d = mujoco.MjData(self.mjm)

    def _np_step(self, q, v, c):
        q, v, c = np.asarray(q, np.float64), np.asarray(v, np.float64), np.asarray(c, np.float64)
        qo, vo = np.empty_like(q), np.empty_like(v)
        d = self._d
        for i in range(len(q)):
            mujoco.mj_resetData(self.mjm, d)
            d.qpos[:], d.qvel[:], d.ctrl[:] = q[i], v[i], c[i]
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.mjm, d)
            qo[i], vo[i] = d.qpos, d.qvel
        return qo.astype(np.float32), vo.astype(np.float32)

    def __call__(self, q, v, c):
        import jax

        shapes = (jax.ShapeDtypeStruct(q.shape, np.float32), jax.ShapeDtypeStruct(v.shape, np.float32))
        return jax.pure_callback(self._np_step, shapes, q, v, c)
