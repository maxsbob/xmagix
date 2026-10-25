"""Correctness tests: CPU/GPU physics parity, determinism, observation/action shapes, task
functions on hand-built states, auto-reset, the Gymnasium view and data files."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xmagix
from xmagix.tasks import TASKS

GPU = any(d.platform == "gpu" for d in jax.devices())
needs_gpu = pytest.mark.skipif(not GPU, reason="MJWarp backend needs a GPU")


@needs_gpu
@pytest.mark.parametrize("emb", list(xmagix.EMBODIMENTS))
@pytest.mark.parametrize("task,steps,atol", [("reach", 40, 5e-3), ("push", 10, 2e-2)])
def test_warp_matches_cpu_mujoco(emb, task, steps, atol):
    """Same start, same actions: MJWarp and CPU MuJoCo agree. Free motion is checked over 40
    control steps; with contacts (chaotic over long horizons) over 10."""
    n = 4
    gpu = xmagix.make(task, emb, n, auto_reset=False)
    cpu = xmagix.make(task, emb, n, backend="cpu", auto_reset=False)
    sg = sc = gpu.reset(jax.random.PRNGKey(3))
    for _ in range(steps):
        a = gpu.expert(sg)
        sg, _, _ = gpu.step(sg, a)
        sc, _, _ = cpu.step(sc, a)
    np.testing.assert_allclose(np.asarray(sg.q), np.asarray(sc.q), atol=atol)


@needs_gpu
def test_reproducible():
    """Same key, same trajectory up to float atomics in MJWarp's contact solver (not bit-exact)."""
    env = xmagix.make("sweep", "gripper", 16)
    k = jax.random.PRNGKey(0)
    pol = lambda o, s, kk: env.expert(s, kk, 0.3)  # noqa: E731
    _, a = xmagix.rollout(env, pol, k, 50)
    _, b = xmagix.rollout(env, pol, k, 50)
    np.testing.assert_allclose(np.asarray(a["q"]), np.asarray(b["q"]), atol=1e-2)


@needs_gpu
@pytest.mark.parametrize("emb", list(xmagix.EMBODIMENTS))
def test_shapes_and_autoreset(emb):
    env = xmagix.make("reach", emb, 8)
    s = env.reset(jax.random.PRNGKey(0))
    assert env.observe(s).shape == (8, env.obs_dim)
    for t in range(env.horizon + 1):
        s, obs, info = env.step(s, jnp.zeros((8, env.act_dim)), jax.random.PRNGKey(t))
        if t == env.horizon - 1:
            assert bool(info["done"].all()) and int(s.t.max()) == 0  # reset happened
    assert obs.shape == (8, env.obs_dim) and np.isfinite(np.asarray(obs)).all()


def test_success_functions_on_hand_built_states():
    eff = jnp.zeros((1, 2))
    ts = jnp.zeros((1, 1))
    push = TASKS["push"]
    at_goal = jnp.asarray([[[*push.goal, 0.0]]])
    assert bool(push.success(at_goal, eff, ts)[0])
    assert not bool(push.success(at_goal + 0.2, eff, ts)[0])
    sweep = TASKS["sweep"]
    up = jnp.asarray([[[0.0, 0.6, 0], [0.3, 0.5, 0], [-0.3, 0.9, 0]]])
    assert bool(sweep.success(up, eff, ts)[0]) and not bool(sweep.success(up.at[0, 1, 1].set(0.0), eff, ts)[0])
    match = TASKS["match"]
    cx, cy = match.region[:2]
    inside = jnp.asarray([[[cx, cy, 0], [cx + 0.1, cy, 0], [0.6, -0.6, 0], [0.6, 0.6, 0]]])
    assert bool(match.success(inside, eff, ts)[0])
    assert not bool(match.success(inside.at[0, 2, :2].set(jnp.asarray([cx, cy - 0.1])), eff, ts)[0])  # distractor in
    route = TASKS["route_left"]
    blk = jnp.asarray([[[*route.goal, 0.0]]])
    assert not bool(route.success(blk, eff, ts)[0])  # at the goal but not through the left gap
    ts2 = route.update(ts, jnp.asarray([[[route.gate[0], route.gate[1], 0.0]]]), eff, eff)
    assert bool(route.success(blk, eff, ts2)[0])
    orbit = TASKS["orbit_ccw"]
    c = jnp.asarray(orbit.pillars[0][:2])
    t = jnp.zeros((1, 1))
    for ang in np.linspace(0, 2 * np.pi + 0.2, 60):
        prev = c + 0.35 * jnp.asarray([[np.cos(ang - 0.11), np.sin(ang - 0.11)]])
        cur = c + 0.35 * jnp.asarray([[np.cos(ang), np.sin(ang)]])
        t = orbit.update(t, None, cur, prev)
    assert bool(orbit.success(None, eff, t)[0]) and not bool(TASKS["orbit_cw"].success(None, eff, t)[0])


@needs_gpu
def test_gymnasium_view():
    from xmagix.vector import make_vec

    venv = make_vec("push", "point", 4)
    obs, _ = venv.reset(seed=0)
    assert obs.shape == (4, venv.env.obs_dim)
    obs, r, term, trunc, info = venv.step(np.zeros((4, 2), np.float32))
    assert obs.shape == (4, venv.env.obs_dim) and r.shape == (4,) and "success" in info


@needs_gpu
def test_play_data_and_demos(tmp_path):
    from xmagix.datasets import collect, demonstrations

    env = xmagix.make("push", "point", 32)
    data = collect(env, jax.random.PRNGKey(0), 100)
    assert data["observations"].shape == (3200, env.obs_dim) and data["terminals"].sum() >= 32
    obs, qs, truncated = demonstrations(xmagix.make("push", "point", 16), jax.random.PRNGKey(1), 3)
    assert len(obs) == 3 and not truncated and all(o.shape[1] == env.obs_dim for o in obs)


@needs_gpu
@pytest.mark.parametrize("emb", list(xmagix.EMBODIMENTS))
def test_render_matches_physics(emb):
    """The drawing uses planar kinematics from the model: body poses must equal MuJoCo's."""
    import mujoco

    from xmagix.draw import body_poses

    env = xmagix.make("push", emb, 2, auto_reset=False)
    s = env.reset(jax.random.PRNGKey(0))
    for _ in range(5):
        s, _, _ = env.step(s, env.expert(s))
    poses = body_poses(env.mjm, s.q)
    d = mujoco.MjData(env.mjm)
    d.qpos[:] = np.asarray(s.q[1], np.float64)
    mujoco.mj_kinematics(env.mjm, d)
    for b in range(1, env.mjm.nbody):
        x, y, a = (float(t[1]) for t in poses[b])
        np.testing.assert_allclose([x, y], d.xpos[b][:2], atol=1e-4)
        R = d.xmat[b].reshape(3, 3)
        assert abs(np.angle(np.exp(1j * (a - np.arctan2(R[1, 0], R[0, 0]))))) < 1e-4


@needs_gpu
def test_pixel_observations():
    env = xmagix.make("sweep", "gripper", 8, obs_type="pixels", pixel_size=64)
    s = env.reset(jax.random.PRNGKey(0))
    obs = env.observe(s)
    assert obs.shape == (8, 64, 64, 3) and obs.dtype == jnp.uint8
    s2, obs2, _ = env.step(s, env.expert(s), jax.random.PRNGKey(1))
    assert obs2.shape == (8, 64, 64, 3)
    img = np.asarray(obs[0]).astype(int)
    assert img.std() > 5 and (img[..., 0] - img[..., 1]).max() > 40  # pink blocks / zone present
