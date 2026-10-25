"""Tasks: objects, fixed goal regions, initial-state distributions, success and the
embodiment-independent expert plan (where the effector should go next).

Success depends only on object states (and, for Reach / Orbit, on the effector path), so one
demonstration is a valid specification for every embodiment. Arena: [-1, 1]^2.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

BLOCK = 0.06  # half side of a block
COLOURS = {"red": "0.85 0.25 0.2 1", "green": "0.2 0.7 0.3 1", "yellow": "0.9 0.75 0.15 1", "grey": "0.5 0.5 0.5 1"}


@dataclass(frozen=True)
class Task:
    name: str
    horizon: int
    blocks: tuple = ()  # colours
    agent_start: tuple = ((0.0, -0.6), 0.1)  # (centre, half-width) of the agent's start box
    block_starts: tuple = ()  # per block (centre, half-width)
    obstacles: tuple = ()  # (x, y, half-x, half-y) static boxes
    pillars: tuple = ()  # (x, y, radius) static cylinders
    n_task_state: int = 1

    # ---- implemented per task
    def success(self, blocks, eff, ts):
        raise NotImplementedError

    def update(self, ts, blocks, eff, eff_prev):
        return ts

    def plan(self, blocks, eff, ts, contact):
        """-> (effector target (N,2), push direction or heading (N,), is_push (N,)) ."""
        raise NotImplementedError

    def score(self, blocks, eff, ts):
        return self.success(blocks, eff, ts).astype(jnp.float32)

    def init_blocks(self, key, n):
        if not self.blocks:
            return jnp.zeros((n, 0, 3))
        ks = jax.random.split(key, 2 * len(self.blocks))
        out = []
        for i, (c, w) in enumerate(self.block_starts):
            xy = jnp.asarray(c) + w * jax.random.uniform(ks[2 * i], (n, 2), minval=-1, maxval=1)
            yaw = jax.random.uniform(ks[2 * i + 1], (n, 1), minval=-jnp.pi / 4, maxval=jnp.pi / 4)
            out.append(jnp.concatenate([xy, yaw], -1))
        return jnp.stack(out, 1)  # (n, B, 3)


def push_plan(b, goal, eff, contact):
    """Push block centre `b` towards `goal`. `contact` = (offset, width): the effector's
    contact surface lies `offset` ahead of its reference point, and it can push up to `width`
    off-centre. Stateless: go round the block if not behind it, line up, then push through.
    Returns (target, push direction angle, pushing flag)."""
    offset, width, body = contact
    d = goal - b
    dist = jnp.linalg.norm(d, axis=-1, keepdims=True)
    u = d / jnp.maximum(dist, 1e-6)
    perp = jnp.stack([-u[:, 1], u[:, 0]], -1)
    rel = eff - b
    along, lat = (rel * u).sum(-1), (rel * perp).sum(-1)
    clear = 1.42 * BLOCK + offset  # no contact beyond this distance, whatever the block's yaw
    around = 1.42 * BLOCK + body  # ... for the whole body (sticks are wide) when passing beside it
    behind = b - u * (clear + 0.03)
    back = 0.75 * BLOCK + 0.5 * offset
    on_back = along < -back
    # beside the block and already slightly behind it, so reaching it switches to "line up"
    side = b + jnp.where(lat[:, None] >= 0, 1.0, -1.0) * perp * (around + 0.05) - u * (back + 0.04)
    lined = jnp.abs(lat) < width
    blocked = jnp.abs(lat) < around + 0.02  # a straight move back would hit the block
    push = b + u * jnp.minimum(0.25, dist + 0.02) - perp * lat[:, None]
    target = jnp.where((on_back & lined)[:, None], push,
                       jnp.where((~on_back & blocked)[:, None], side, behind))
    arrived = dist[:, 0] < 0.04  # hold still once the block is at its goal
    target = jnp.where(arrived[:, None], eff, target)
    face = jnp.where(arrived[:, None], -rel, u)  # when holding, keep facing the block
    return target, jnp.arctan2(face[:, 1], face[:, 0]), on_back & lined


@dataclass(frozen=True)
class PushToRegion(Task):
    goal: tuple = (0.45, 0.35)
    radius: float = 0.12

    def success(self, blocks, eff, ts):
        return jnp.linalg.norm(blocks[:, 0, :2] - jnp.asarray(self.goal), axis=-1) < self.radius

    def plan(self, blocks, eff, ts, contact=(0.06, 0.035, 0.06)):
        return push_plan(blocks[:, 0, :2], jnp.broadcast_to(jnp.asarray(self.goal), eff.shape), eff, contact)


@dataclass(frozen=True)
class SweepToTop(Task):
    """x-magical's SweepToTop: every debris block above the line y = `line`."""

    line: float = 0.35

    def success(self, blocks, eff, ts):
        return jnp.all(blocks[:, :, 1] > self.line, -1)

    def score(self, blocks, eff, ts):
        return jnp.mean(blocks[:, :, 1] > self.line, -1)

    def plan(self, blocks, eff, ts, contact=(0.06, 0.035, 0.06)):
        # blocks in a fixed order (index), so the choice does not flip while a block moves
        done = blocks[:, :, 1] > self.line + 0.1
        i = jnp.argmin(done, -1)
        b = jnp.take_along_axis(blocks[:, :, :2], i[:, None, None], 1)[:, 0]
        goal = jnp.stack([jnp.clip(b[:, 0], -0.5, 0.5), jnp.full(b.shape[:1], self.line + 0.17)], -1)
        return push_plan(b, goal, eff, contact)


@dataclass(frozen=True)
class MatchRegions(Task):
    """MatchRegions (from MAGICAL): all blocks of the target colour inside the region, none of the
    others. Blocks 0..k-1 are targets."""

    n_target: int = 2
    region: tuple = (-0.45, 0.3, 0.25, 0.2)  # centre x, y, half sizes

    def inside(self, blocks):
        cx, cy, hx, hy = self.region
        return (jnp.abs(blocks[:, :, 0] - cx) < hx) & (jnp.abs(blocks[:, :, 1] - cy) < hy)

    def success(self, blocks, eff, ts):
        ins = self.inside(blocks)
        return jnp.all(ins[:, : self.n_target], -1) & ~jnp.any(ins[:, self.n_target:], -1)

    def plan(self, blocks, eff, ts, contact=(0.06, 0.035, 0.06)):
        ins = self.inside(blocks)[:, : self.n_target]
        i = jnp.argmin(ins, -1)  # first target block outside the region
        b = jnp.take_along_axis(blocks[:, :, :2], i[:, None, None], 1)[:, 0]
        goal = jnp.broadcast_to(jnp.asarray(self.region[:2]), b.shape)
        return push_plan(b, goal, eff, contact)


@dataclass(frozen=True)
class Route(Task):
    """Push the block to the goal *through the demonstrated gate* of a wall with two gaps:
    goal reaching alone does not specify the task (path dependence)."""

    goal: tuple = (0.0, 0.45)
    radius: float = 0.12
    gate: tuple = (-0.55, 0.05)  # centre of the gap to use
    gap_half: float = 0.35  # half width of each gap

    def update(self, ts, blocks, eff, eff_prev):
        b = blocks[:, 0, :2]
        through = (jnp.abs(b[:, 0] - self.gate[0]) < self.gap_half) & (jnp.abs(b[:, 1] - self.gate[1]) < 0.1)
        return ts.at[:, 0].set(jnp.maximum(ts[:, 0], through.astype(ts.dtype)))

    def success(self, blocks, eff, ts):
        at_goal = jnp.linalg.norm(blocks[:, 0, :2] - jnp.asarray(self.goal), axis=-1) < self.radius
        return at_goal & (ts[:, 0] > 0)

    def plan(self, blocks, eff, ts, contact=(0.06, 0.035, 0.06)):
        b = blocks[:, 0, :2]
        gx, gy = self.gate
        below = jnp.asarray([gx, gy - 0.3])
        above = jnp.asarray([gx, gy + 0.25])
        lined = jnp.abs(b[:, :1] - gx) < 0.08
        goal = jnp.where(ts[:, :1] > 0, jnp.where(b[:, 1:] > gy + 0.15, jnp.asarray(self.goal), above),
                         jnp.where(lined | (b[:, 1:] > gy - 0.2), above, below))
        return push_plan(b, goal, eff, contact)


@dataclass(frozen=True)
class Orbit(Task):
    """Circle the pillar with the effector in the demonstrated direction (laps are signed):
    state-occupancy matching cannot tell the two directions apart."""

    direction: float = 1.0  # +1 counter-clockwise
    orbit_radius: float = 0.35

    def angle(self, eff):
        c = jnp.asarray(self.pillars[0][:2])
        return jnp.arctan2(eff[:, 1] - c[1], eff[:, 0] - c[0])

    def update(self, ts, blocks, eff, eff_prev):
        d = (self.angle(eff) - self.angle(eff_prev) + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return ts.at[:, 0].add(d / (2 * jnp.pi))

    def success(self, blocks, eff, ts):
        return self.direction * ts[:, 0] >= 1.0

    def score(self, blocks, eff, ts):
        return self.direction * ts[:, 0]  # laps

    def plan(self, blocks, eff, ts, contact=(0.0, 0.035, 0.06)):
        c = jnp.asarray(self.pillars[0][:2])
        a = self.angle(eff) + self.direction * 0.7
        target = c + self.orbit_radius * jnp.stack([jnp.cos(a), jnp.sin(a)], -1)
        heading = a + self.direction * jnp.pi / 2
        return target, heading, jnp.zeros(eff.shape[0], bool)


@dataclass(frozen=True)
class Reach(Task):
    """Visit the waypoints in order (no objects): calibrates alignment without manipulation."""

    waypoints: tuple = ((-0.5, -0.2), (0.0, 0.3), (0.5, -0.2))
    radius: float = 0.1

    def update(self, ts, blocks, eff, eff_prev):
        wp = jnp.asarray(self.waypoints)
        k = ts[:, 0].astype(jnp.int32)
        cur = wp[jnp.minimum(k, len(self.waypoints) - 1)]
        hit = (jnp.linalg.norm(eff - cur, axis=-1) < self.radius) & (k < len(self.waypoints))
        return ts.at[:, 0].add(hit.astype(ts.dtype))

    def success(self, blocks, eff, ts):
        return ts[:, 0] >= len(self.waypoints)

    def score(self, blocks, eff, ts):
        return ts[:, 0] / len(self.waypoints)

    def plan(self, blocks, eff, ts, contact=(0.0, 0.035, 0.06)):
        wp = jnp.asarray(self.waypoints)
        target = wp[jnp.minimum(ts[:, 0].astype(jnp.int32), len(self.waypoints) - 1)]
        d = target - eff
        return target, jnp.arctan2(d[:, 1], d[:, 0]), jnp.zeros(eff.shape[0], bool)


TASKS = {
    "reach": Reach("reach", horizon=200, agent_start=((0.0, -0.5), 0.1)),
    "push": PushToRegion("push", horizon=300, blocks=("red",), agent_start=((-0.5, -0.6), 0.1),
                         block_starts=(((-0.1, -0.2), 0.08),)),
    "sweep": SweepToTop("sweep", horizon=500, blocks=("red", "red", "red"), agent_start=((0.0, -0.75), 0.1),
                        block_starts=(((-0.45, -0.3), 0.1), ((0.0, -0.2), 0.1), ((0.45, -0.3), 0.1))),
    "match": MatchRegions("match", horizon=500, blocks=("green", "green", "red", "red"),
                          agent_start=((0.4, -0.7), 0.1),
                          block_starts=(((0.0, -0.2), 0.08), ((0.4, 0.05), 0.08), ((-0.55, -0.35), 0.08),
                                        ((0.45, 0.5), 0.08))),
    "route_left": Route("route_left", horizon=400, blocks=("red",), agent_start=((0.0, -0.75), 0.08),
                        block_starts=(((0.0, -0.45), 0.06),),
                        obstacles=((0.0, 0.05, 0.2, 0.04), (-1.0, 0.05, 0.1, 0.04), (1.0, 0.05, 0.1, 0.04)),
                        gate=(-0.55, 0.05)),
    "route_right": Route("route_right", horizon=400, blocks=("red",), agent_start=((0.0, -0.75), 0.08),
                         block_starts=(((0.0, -0.45), 0.06),),
                         obstacles=((0.0, 0.05, 0.2, 0.04), (-1.0, 0.05, 0.1, 0.04), (1.0, 0.05, 0.1, 0.04)),
                         gate=(0.55, 0.05)),
    "orbit_ccw": Orbit("orbit_ccw", horizon=300, agent_start=((0.3, -0.6), 0.05), pillars=((0.0, -0.25, 0.12),),
                       direction=1.0),
    "orbit_cw": Orbit("orbit_cw", horizon=300, agent_start=((-0.3, -0.6), 0.05), pillars=((0.0, -0.25, 0.12),),
                      direction=-1.0),
}
