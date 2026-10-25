"""Batched, anti-aliased drawing of states in pure JAX (signed-distance rendering).

Shapes and poses come from the MuJoCo model itself: planar forward kinematics over its bodies
and joints, then every geom is drawn with its exact shape. So the picture always matches the
physics, for any embodiment. Styling follows x-magical (grey googly-eyed robots, pastel goal
zones), with outlines, rounded blocks, soft shadows, a faint floor grid, and pupils that look
where the robot is heading.

    img = render(env, state.q, state.v, size=96)          # (N, 96, 96, 3) uint8, jit-able
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

EXTENT = 1.12  # half width of the drawn square (walls end at 1.1)


def _hex(h, a=1.0):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], np.float32)


PALETTE = {
    "floor": _hex("#fbfbfd"), "grid": _hex("#e4e7ef"), "wall": _hex("#dfe2ea"), "wall_edge": _hex("#a9aebd"),
    "zone_goal": _hex("#f4c6d6", 0.85), "zone_region": _hex("#cfeedd", 0.9),
    "robot": _hex("#a9acb9"), "robot_edge": _hex("#5f6272"), "part": _hex("#dcdee6"), "part_edge": _hex("#7d8192"),
    "eye": _hex("#ffffff"), "eye_edge": _hex("#5f6272"), "pupil": _hex("#18181d"),
    "shadow": _hex("#1e2332", 0.16), "trail": _hex("#5f6272", 0.55),
    "red": (_hex("#ec8aa6"), _hex("#bf5a79")), "green": (_hex("#7fcfa0"), _hex("#4c9a6c")),
    "yellow": (_hex("#f5cf5a"), _hex("#c79f22")), "grey": (_hex("#c3c7d2"), _hex("#8a8fa0")),
}
BLOCK_RGBA = {"red": (0.85, 0.25, 0.2), "green": (0.2, 0.7, 0.3), "yellow": (0.9, 0.75, 0.15), "grey": (0.5, 0.5, 0.5)}


# ------------------------------------------------------------------------------ signed distances
def sd_circle(p, c, r):
    return jnp.linalg.norm(p - c, axis=-1) - r


def sd_box(p, c, ang, h, rr):
    """Rounded box centred at c, rotated by ang, half extents h, corner radius rr."""
    d = p - c
    cs, sn = jnp.cos(ang)[..., None], jnp.sin(ang)[..., None]
    local = jnp.stack([cs[..., 0] * d[..., 0] + sn[..., 0] * d[..., 1],
                       -sn[..., 0] * d[..., 0] + cs[..., 0] * d[..., 1]], -1)
    q = jnp.abs(local) - (jnp.asarray(h) - rr)
    return jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, -1), 0.0) - rr


def sd_capsule(p, a, b, r):
    pa, ba = p - a, b - a
    t = jnp.clip((pa * ba).sum(-1) / jnp.maximum((ba * ba).sum(-1), 1e-9), 0.0, 1.0)
    return jnp.linalg.norm(pa - t[..., None] * ba, axis=-1) - r


# ------------------------------------------------------------------------------ scene from model
@dataclass(frozen=True)
class Geom:
    kind: str  # circle | box | capsule
    body: int
    pos: tuple  # in the body frame
    ang: float  # box yaw / capsule axis angle in the body frame
    size: tuple  # box half extents | (radius,) | capsule (radius, half length)
    role: str  # zone_goal | zone_region | wall | block:<colour> | robot | part
    shadow: bool


def _yaw(quat):
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def scene(mjm):
    """Static description of what to draw, read from the compiled MuJoCo model."""
    out = []
    for g in range(mjm.ngeom):
        t = int(mjm.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        b = int(mjm.geom_bodyid[g])
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        rgba = mjm.geom_rgba[g]
        if mjm.geom_contype[g] == 0 and mjm.geom_conaffinity[g] == 0:
            role = "zone_region" if t == int(mujoco.mjtGeom.mjGEOM_BOX) and rgba[3] > 0.3 else "zone_goal"
        elif b == 0:
            role = "wall"
        elif name.startswith("block"):
            col = min(BLOCK_RGBA, key=lambda k: np.abs(np.asarray(BLOCK_RGBA[k]) - rgba[:3]).sum())
            role = f"block:{col}"
        else:
            role = "robot" if rgba[2] > 0.8 else "part"
        pos = tuple(float(x) for x in mjm.geom_pos[g][:2])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, mjm.geom_quat[g])
        R = R.reshape(3, 3)
        size = mjm.geom_size[g]
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            out.append(Geom("box", b, pos, _yaw(mjm.geom_quat[g]), (float(size[0]), float(size[1])), role, role != "wall" and not role.startswith("zone")))
        elif t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE)):
            out.append(Geom("circle", b, pos, 0.0, (float(size[0]),), role, role != "wall" and not role.startswith("zone")))
        elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            out.append(Geom("capsule", b, pos, float(np.arctan2(R[1, 2], R[0, 2])), (float(size[0]), float(size[1])),
                            role, True))
    order = {"zone_goal": 0, "zone_region": 0, "wall": 1, "part": 3, "robot": 4}
    return sorted(out, key=lambda g: order.get(g.role, 2))


def body_poses(mjm, q):
    """Planar forward kinematics: (x, y, yaw) of every body, batched over worlds. Joints are
    slides in the body plane and hinges about z, applied in the body frame in model order."""
    n = q.shape[0]
    zeros = jnp.zeros(n)
    poses = [(zeros, zeros, zeros)]
    for b in range(1, mjm.nbody):
        px, py, pa = poses[int(mjm.body_parentid[b])]
        bx, by = (float(v) for v in mjm.body_pos[b][:2])
        ba = _yaw(mjm.body_quat[b])
        c, s = jnp.cos(pa), jnp.sin(pa)
        x, y, a = px + c * bx - s * by, py + s * bx + c * by, pa + ba
        for j in range(int(mjm.body_jntadr[b]), int(mjm.body_jntadr[b]) + int(mjm.body_jntnum[b])):
            qj = q[:, int(mjm.jnt_qposadr[j])]
            ax = mjm.jnt_axis[j]
            if int(mjm.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                c, s = jnp.cos(a), jnp.sin(a)
                x, y = x + (c * ax[0] - s * ax[1]) * qj, y + (s * ax[0] + c * ax[1]) * qj
            else:  # hinge about +-z
                a = a + float(np.sign(ax[2])) * qj
        poses.append((x, y, a))
    return poses


# ------------------------------------------------------------------------------ rendering
def _over(img, colour, alpha):
    """Composite `colour` (rgba) with per-pixel coverage `alpha` (N, H, W) over img (N, H, W, 3)."""
    a = (alpha * colour[3])[..., None]
    return img * (1 - a) + colour[:3] * a


def _shape_sd(g, poses, p, grow=0.0, offset=(0.0, 0.0)):
    x, y, a = poses[g.body]
    c, s = jnp.cos(a), jnp.sin(a)
    cx = x + c * g.pos[0] - s * g.pos[1] + offset[0]
    cy = y + s * g.pos[0] + c * g.pos[1] + offset[1]
    centre = jnp.stack([cx, cy], -1)[:, None, None, :]
    pp = p[None]
    if g.kind == "circle":
        return sd_circle(pp, centre, g.size[0] + grow)
    if g.kind == "box":
        rr = {"wall": 0.0, "zone_goal": 0.03, "zone_region": 0.03}.get(
            g.role, min(g.size) * (0.3 if g.role.startswith("block") else 0.6))
        return sd_box(pp, centre, (a + g.ang)[:, None, None], jnp.asarray(g.size) + grow, rr + grow)
    ang = a + g.ang
    d = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)[:, None, None, :] * g.size[1]
    return sd_capsule(pp, centre - d, centre + d, g.size[0] + grow)


def render(env, q, v=None, size=96, supersample=1, trail=None):
    """Images (N, size, size, 3) uint8 of the worlds in `q` (and velocities `v`, which point the
    eyes of bodies without a heading). `trail`: optional (N, T, 2) effector positions."""
    mjm, emb = env.mjm, env.emb
    res = size * supersample
    px = 2 * EXTENT / res
    coords = (jnp.arange(res) + 0.5) * px - EXTENT
    X, Y = jnp.meshgrid(coords, -coords)  # row 0 = top of the arena
    p = jnp.stack([X, Y], -1)
    n = q.shape[0]
    poses = body_poses(mjm, q)
    aa = lambda d: jnp.clip(0.5 - d / px, 0.0, 1.0)  # noqa: E731  (anti-aliased coverage)
    edge = max(1.2 * px, 0.007)
    img = jnp.broadcast_to(PALETTE["floor"][:3], (n, res, res, 3))
    # distance to the nearest grid line (every 0.25 units)
    grid = 0.125 - jnp.maximum(jnp.abs((X + 1e-3) % 0.25 - 0.125), jnp.abs((Y + 1e-3) % 0.25 - 0.125))
    img = _over(img, PALETTE["grid"], jnp.broadcast_to(jnp.clip(1.0 - grid / (0.8 * px), 0, 1), (n, res, res)))
    geoms = scene(mjm)
    for g in geoms:  # zones under everything
        if g.role.startswith("zone"):
            img = _over(img, PALETTE[g.role], aa(_shape_sd(g, poses, p)))
    for g in geoms:  # soft shadows of everything that moves
        if g.shadow:
            d = _shape_sd(g, poses, p, offset=(0.014, -0.018))
            img = _over(img, PALETTE["shadow"], jnp.clip(0.5 - d / 0.03, 0.0, 1.0))
    if trail is not None:
        T = trail.shape[1]
        for t in range(0, T, max(1, T // 60)):
            col = PALETTE["trail"].copy()
            col[3] *= 0.15 + 0.85 * t / T
            img = _over(img, col, aa(sd_circle(p[None], trail[:, t][:, None, None, :], 0.007)))
    for g in geoms:
        if g.role.startswith("zone"):
            continue
        if g.role == "wall":
            fill, line = PALETTE["wall"], PALETTE["wall_edge"]
        elif g.role.startswith("block"):
            fill, line = PALETTE[g.role.split(":")[1]]
        else:
            fill, line = (PALETTE["robot"], PALETTE["robot_edge"]) if g.role == "robot" else (PALETTE["part"], PALETTE["part_edge"])
        d = _shape_sd(g, poses, p)
        img = _over(img, line, aa(d))
        img = _over(img, fill, aa(d + edge))
    img = _eyes(img, env, poses, q, v, p, aa, edge)
    if supersample > 1:
        img = img.reshape(n, size, supersample, size, supersample, 3).mean((2, 4))
    return (jnp.clip(img, 0, 1) * 255).astype(jnp.uint8)


def _eyes(img, env, poses, q, v, p, aa, edge):
    emb = env.emb
    body, (ex, ey), sep, r = emb.eyes
    bi = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_BODY, body)
    x, y, a = poses[bi]
    c, s = jnp.cos(a), jnp.sin(a)
    centre = jnp.stack([x + c * ex - s * ey, y + s * ex + c * ey], -1)
    if emb.oriented or body != "agent":
        gaze = jnp.stack([c, s], -1)
    else:  # a disc has no heading: look where it moves
        vel = v[:, :2] if v is not None else jnp.zeros((q.shape[0], 2))
        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        gaze = jnp.where(speed > 0.02, vel / jnp.maximum(speed, 1e-6), jnp.asarray([0.0, 1.0]))
    side = jnp.stack([-gaze[:, 1], gaze[:, 0]], -1)
    for sgn in (1.0, -1.0):
        e = centre + sgn * sep * side
        d = sd_circle(p[None], e[:, None, None, :], r)
        img = _over(img, PALETTE["eye_edge"], aa(d))
        img = _over(img, PALETTE["eye"], aa(d + 0.6 * edge))
        pupil = e + 0.42 * r * gaze
        img = _over(img, PALETTE["pupil"], aa(sd_circle(p[None], pupil[:, None, None, :], 0.5 * r)))
    return img
