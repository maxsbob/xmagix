"""Top-down drawings of states and trajectories (matplotlib), from MuJoCo's own geom poses, so
every embodiment and task is drawn exactly as simulated."""

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Circle, Polygon


def draw_state(ax, mjm, qpos, alpha=1.0, regions=True):
    d = mujoco.MjData(mjm)
    d.qpos[:] = np.asarray(qpos, np.float64)
    mujoco.mj_kinematics(mjm, d)
    for g in range(mjm.ngeom):
        t = int(mjm.geom_type[g])  # numpy ints do not compare equal to the pybind enums
        rgba = mjm.geom_rgba[g].copy()
        if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        region = mjm.geom_contype[g] == 0 and mjm.geom_conaffinity[g] == 0
        if region and not regions:
            continue
        rgba[3] *= alpha
        pos, R, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), mjm.geom_size[g]
        z = 0 if region else 2
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size[:2]
            ax.add_patch(Polygon(pos[:2] + corners @ R[:2, :2].T, color=rgba, zorder=z, lw=0))
        elif t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE)):
            ax.add_patch(Circle(pos[:2], size[0], color=rgba, zorder=z, lw=0))
        elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            axis, r = R[:2, 2] * size[1], size[0]
            n = np.array([-axis[1], axis[0]]) / (np.linalg.norm(axis) + 1e-9) * r
            ax.add_patch(Polygon([pos[:2] - axis + n, pos[:2] + axis + n, pos[:2] + axis - n, pos[:2] - axis - n],
                                 color=rgba, zorder=z, lw=0))
            ax.add_patch(Circle(pos[:2] - axis, r, color=rgba, zorder=z, lw=0))
            ax.add_patch(Circle(pos[:2] + axis, r, color=rgba, zorder=z, lw=0))
    ax.set_xlim(-1.12, 1.12), ax.set_ylim(-1.12, 1.12), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])


def draw_episode(ax, mjm, qs, effector=None, every=40, title=None):
    """Faded snapshots of an episode plus the effector path."""
    qs = np.asarray(qs)
    for i in range(0, len(qs), every):
        draw_state(ax, mjm, qs[i], alpha=0.25 + 0.5 * i / len(qs), regions=i == 0)
    draw_state(ax, mjm, qs[-1], regions=False)
    if effector is not None:
        e = np.asarray(effector)
        ax.plot(e[:, 0], e[:, 1], color="k", lw=0.8, alpha=0.7, zorder=3)
    if title:
        ax.set_title(title, fontsize=8)


def episode_grid(envs_and_qs, path, cols=4):
    rows = (len(envs_and_qs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (env, qs, eff, title) in zip(axes.flat, envs_and_qs):
        ax.axis("on")
        draw_episode(ax, env.mjm, qs, eff, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
