"""Embodiments: MJCF body + actuators, action -> control map, proprioception, effector, and a
scripted low-level controller that drives the effector to a target (used by the experts).

All bodies live in one top-down plane (x, y, yaw); z is fixed. An embodiment's joints come first
in qpos, so its state is q[:, :nq] / v[:, :nq]. Functions take and return JAX arrays batched
over worlds.
"""

from dataclasses import dataclass

import jax.numpy as jnp

Z = 0.05  # height of every geom centre; nothing moves in z
AGENT = 'contype="1" conaffinity="3"'  # collides with blocks (1) and walls (2)
ARM = 'contype="4" conaffinity="1"'  # collides with blocks only (fixed base reaches over walls)


def rot(yaw, xy):
    """Rotate world vectors (..., 2) into the frame with heading `yaw` (inverse rotation)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1]], -1)


def unrot(yaw, xy):
    """Rotate body-frame vectors (..., 2) into the world frame."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1)


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


@dataclass(frozen=True)
class Embodiment:
    name: str
    nq: int
    act_dim: int
    push_offset: float  # effector reference point -> contact surface, along the push direction
    oriented: bool = True  # has a heading the expert controls
    push_width: float = 0.035  # how far off-centre it can push a block and still move it straight
    body_radius: float = 0.06  # clearance its body needs when passing beside a block
    time_scale: float = 1.0  # episode horizons are multiplied by this (slow bodies get more time)

    @property
    def contact(self):
        return self.push_offset, self.push_width, self.body_radius

    def xml(self) -> tuple[str, str]:  # (worldbody snippet, actuator snippet)
        raise NotImplementedError

    def ctrl(self, q, v, a):
        """Actions in [-1, 1] -> actuator controls."""
        return a

    def pose(self, q):
        """(xy, yaw) of the frame observations are expressed in."""
        raise NotImplementedError

    def effector(self, q):
        raise NotImplementedError

    def proprio(self, q, v):
        raise NotImplementedError

    def track(self, q, v, target, heading):
        """Action moving the effector to `target` (N, 2), facing `heading` (N,) if oriented."""
        raise NotImplementedError

    def init(self, key, n, centre, spread):
        """Initial qpos (n, nq) around `centre`."""
        raise NotImplementedError


# ------------------------------------------------------------------------------------- disc
@dataclass(frozen=True)
class Point(Embodiment):
    """Holonomic disc, world-frame force control (x-magical has no such body; it is the reference)."""

    radius: float = 0.06
    gear: float = 2.0
    mass: float = 1.0
    damping: float = 4.0

    def xml(self):
        body = f"""
    <body name="agent" pos="0 0 {Z}">
      <joint name="ax" type="slide" axis="1 0 0" damping="{self.damping}"/>
      <joint name="ay" type="slide" axis="0 1 0" damping="{self.damping}"/>
      <geom type="cylinder" size="{self.radius} 0.04" mass="{self.mass}" rgba="0.2 0.4 0.9 1" {AGENT}/>
    </body>"""
        act = f"""
    <motor joint="ax" gear="{self.gear}" ctrlrange="-1 1"/>
    <motor joint="ay" gear="{self.gear}" ctrlrange="-1 1"/>"""
        return body, act

    def pose(self, q):
        return q[:, :2], jnp.zeros(q.shape[0])

    def effector(self, q):
        return q[:, :2]

    def proprio(self, q, v):
        return jnp.concatenate([q[:, :2], v[:, :2]], -1)

    def track(self, q, v, target, heading):
        u = 8.0 * (target - q[:, :2]) - 1.0 * v[:, :2]
        return jnp.clip(u, -1, 1)

    def init(self, key, n, centre, spread):
        import jax

        return centre + spread * jax.random.uniform(key, (n, 2), minval=-1, maxval=1)


# ------------------------------------------------------------------------------- planar rigid body
@dataclass(frozen=True)
class Planar(Embodiment):
    """Free planar body (x, y, yaw) with body-frame actions (forward, lateral, turn)."""

    gear: float = 2.0
    torque: float = 0.4
    mass: float = 1.0
    damping: float = 4.0
    rot_damping: float = 0.4
    turn_first: bool = False  # wide bodies turn in place before translating (else they sweep blocks)

    def joints(self):
        return f"""
      <joint name="ax" type="slide" axis="1 0 0" damping="{self.damping}"/>
      <joint name="ay" type="slide" axis="0 1 0" damping="{self.damping}"/>
      <joint name="ar" type="hinge" axis="0 0 1" damping="{self.rot_damping}"/>"""

    def motors(self):
        return f"""
    <motor joint="ax" gear="{self.gear}" ctrlrange="-1 1"/>
    <motor joint="ay" gear="{self.gear}" ctrlrange="-1 1"/>
    <motor joint="ar" gear="{self.torque}" ctrlrange="-1 1"/>"""

    def ctrl(self, q, v, a):
        return jnp.concatenate([unrot(q[:, 2], a[:, :2]), a[:, 2:3]], -1)

    def pose(self, q):
        return q[:, :2], q[:, 2]

    def proprio(self, q, v):
        yaw = q[:, 2]
        return jnp.concatenate([q[:, :2], jnp.cos(yaw)[:, None], jnp.sin(yaw)[:, None],
                                rot(yaw, v[:, :2]), v[:, 2:3]], -1)

    def body_track(self, q, v, target_body, heading):
        """Move the body origin to `target_body` facing `heading`."""
        u = 8.0 * (target_body - q[:, :2]) - 1.0 * v[:, :2]
        err = wrap(heading - q[:, 2])
        if self.turn_first:
            u = u * jnp.clip(2.0 * jnp.cos(err) - 1.0, 0.0, 1.0)[:, None]
        turn = 4.0 * err - 0.5 * v[:, 2]
        return jnp.clip(jnp.concatenate([rot(q[:, 2], u), turn[:, None]], -1), -1, 1)

    def init(self, key, n, centre, spread):
        import jax

        k1, k2 = jax.random.split(key)
        xy = centre + spread * jax.random.uniform(k1, (n, 2), minval=-1, maxval=1)
        return jnp.concatenate([xy, jax.random.uniform(k2, (n, 1), minval=-jnp.pi, maxval=jnp.pi)], -1)


@dataclass(frozen=True)
class Stick(Planar):
    """A rod pushed broadside (x-magical's short / medium / long sticks). The effector is its
    centre; the expert holds it perpendicular to the push direction."""

    half_length: float = 0.15

    def xml(self):
        L = self.half_length
        body = f"""
    <body name="agent" pos="0 0 {Z}">{self.joints()}
      <geom type="capsule" fromto="0 {-L} 0 0 {L} 0" size="0.025" mass="{self.mass}" rgba="0.2 0.4 0.9 1" {AGENT}/>
    </body>"""
        return body, self.motors()

    def effector(self, q):
        return q[:, :2]

    def track(self, q, v, target, heading):
        # The rod lies along the body y axis, so facing `heading` pushes broadside.
        return self.body_track(q, v, target, heading)


@dataclass(frozen=True)
class Gripper(Planar):
    """Planar parallel gripper: a palm with two sliding fingers (x-magical's gripper). Action:
    (forward, lateral, turn, close). The effector is the point between the fingertips."""

    reach: float = 0.1

    def xml(self):
        body = f"""
    <body name="agent" pos="0 0 {Z}">{self.joints()}
      <geom type="box" size="0.02 0.09 0.04" mass="{self.mass}" rgba="0.2 0.4 0.9 1" {AGENT}/>
      <body name="finger_l" pos="0.05 0.06 0">
        <joint name="fl" type="slide" axis="0 1 0" range="-0.05 0.0" damping="1"/>
        <geom type="box" size="0.05 0.012 0.04" mass="0.1" rgba="0.1 0.25 0.7 1" {AGENT}/>
      </body>
      <body name="finger_r" pos="0.05 -0.06 0">
        <joint name="fr" type="slide" axis="0 1 0" range="0.0 0.05" damping="1"/>
        <geom type="box" size="0.05 0.012 0.04" mass="0.1" rgba="0.1 0.25 0.7 1" {AGENT}/>
      </body>
    </body>"""
        act = self.motors() + """
    <position joint="fl" kp="30" ctrlrange="-0.05 0"/>
    <position joint="fr" kp="30" ctrlrange="0 0.05"/>"""
        return body, act

    def ctrl(self, q, v, a):
        close = 0.025 * (a[:, 3:4] + 1)  # 0 (open) .. 0.05 (closed)
        return jnp.concatenate([super().ctrl(q, v, a[:, :3]), -close, close], -1)

    def effector(self, q):
        return q[:, :2] + unrot(q[:, 2], jnp.tile(jnp.array([[self.reach, 0.0]]), (q.shape[0], 1)))

    def proprio(self, q, v):
        return jnp.concatenate([super().proprio(q, v), (q[:, 4:5] - q[:, 3:4])], -1)

    def track(self, q, v, target, heading):
        # approach point from the *current* heading, so turning does not drag the target around
        base = target - unrot(q[:, 2], jnp.tile(jnp.array([[self.reach, 0.0]]), (q.shape[0], 1)))
        a = self.body_track(q, v, base, heading)
        return jnp.concatenate([a, jnp.ones((q.shape[0], 1))], -1)  # push with closed fingers

    def init(self, key, n, centre, spread):
        return jnp.concatenate([super().init(key, n, centre, spread), jnp.zeros((n, 2))], -1)


@dataclass(frozen=True)
class Scoop(Planar):
    """U-shaped plough: a back plate and two side walls open to the front. It cradles a block
    instead of pushing it off a flat face (different object interaction, same plan). The
    effector is the inside of the U."""

    def xml(self):
        body = f"""
    <body name="agent" pos="0 0 {Z}">{self.joints()}
      <geom type="box" pos="-0.02 0 0" size="0.015 0.11 0.04" mass="{0.6 * self.mass}" rgba="0.2 0.4 0.9 1" {AGENT}/>
      <geom type="box" pos="0.06 0.095 0" size="0.065 0.015 0.04" mass="{0.2 * self.mass}" rgba="0.1 0.25 0.7 1" {AGENT}/>
      <geom type="box" pos="0.06 -0.095 0" size="0.065 0.015 0.04" mass="{0.2 * self.mass}" rgba="0.1 0.25 0.7 1" {AGENT}/>
    </body>"""
        return body, self.motors()

    def effector(self, q):
        return q[:, :2]  # inner face of the back plate is 0.005 ahead of the body origin

    def track(self, q, v, target, heading):
        return self.body_track(q, v, target, heading)


@dataclass(frozen=True)
class DiffDrive(Planar):
    """Non-holonomic cart: action (forward thrust, turn). Wheels are modelled as a lateral
    friction force proportional to sideways slip, so it cannot move sideways."""

    lateral_friction: float = 12.0

    def xml(self):
        body = f"""
    <body name="agent" pos="0 0 {Z}">{self.joints()}
      <geom type="box" size="0.08 0.055 0.04" mass="{self.mass}" rgba="0.2 0.4 0.9 1" {AGENT}/>
      <geom type="box" pos="0.085 0 0" size="0.012 0.07 0.04" mass="0.05" rgba="0.1 0.25 0.7 1" {AGENT}/>
    </body>"""
        motors = self.motors().replace(f'joint="ax" gear="{self.gear}" ctrlrange="-1 1"',
                                       'joint="ax" gear="1" ctrlrange="-20 20"')
        motors = motors.replace(f'joint="ay" gear="{self.gear}" ctrlrange="-1 1"', 'joint="ay" gear="1" ctrlrange="-20 20"')
        return body, motors

    def ctrl(self, q, v, a):
        slip = rot(q[:, 2], v[:, :2])[:, 1]
        body_force = jnp.stack([self.gear * a[:, 0], -self.lateral_friction * slip], -1)
        return jnp.concatenate([unrot(q[:, 2], body_force), a[:, 1:2]], -1)

    def effector(self, q):
        return q[:, :2] + unrot(q[:, 2], jnp.tile(jnp.array([[0.1, 0.0]]), (q.shape[0], 1)))

    def proprio(self, q, v):
        yaw = q[:, 2]
        return jnp.concatenate([q[:, :2], jnp.cos(yaw)[:, None], jnp.sin(yaw)[:, None],
                                rot(yaw, v[:, :2])[:, :1], v[:, 2:3]], -1)

    def track(self, q, v, target, heading):
        # Pure pursuit on the bumper point; the final heading follows from the approach.
        e = self.effector(q)
        d = target - e
        err = wrap(jnp.arctan2(d[:, 1], d[:, 0]) - q[:, 2])
        dist = jnp.linalg.norm(d, axis=-1)
        fwd = jnp.clip(6.0 * dist, 0, 1) * jnp.cos(err).clip(0, 1) - 0.5 * rot(q[:, 2], v[:, :2])[:, 0]
        turn = 3.0 * err - 0.4 * v[:, 2]
        return jnp.clip(jnp.stack([fwd, turn], -1), -1, 1)


# ------------------------------------------------------------------------------------------ arms
@dataclass(frozen=True)
class Arm(Embodiment):
    """Fixed-base planar arm with joint torques (no Cartesian actions); base at the bottom wall.
    Links collide with blocks but not with the walls."""

    links: tuple = (0.6, 0.6)
    base: tuple = (0.0, -1.0)
    torque: float = 1.5
    damping: float = 1.0

    def xml(self):
        bx, by = self.base
        body, close = f'\n    <body name="agent" pos="{bx} {by} {Z}">', ""
        for i, L in enumerate(self.links):
            if i:
                body += f'\n    <body name="link{i}" pos="{self.links[i - 1]} 0 0">'
            body += (f'\n      <joint name="j{i}" type="hinge" axis="0 0 1" damping="{self.damping}"/>'
                     f'\n      <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.03" mass="0.3" rgba="0.2 0.4 0.9 1" {ARM}/>')
            close += "\n    </body>"
        body += f'\n      <geom type="cylinder" pos="{self.links[-1]} 0 0" size="0.045 0.04" mass="0.05" rgba="0.1 0.25 0.7 1" {ARM}/>'
        act = "".join(f'\n    <motor joint="j{i}" gear="{self.torque}" ctrlrange="-1 1"/>' for i in range(len(self.links)))
        return body + close, act

    def angles(self, q):
        return jnp.cumsum(q[:, : self.nq], -1)

    def effector(self, q):
        ang = self.angles(q)
        L = jnp.asarray(self.links)
        return jnp.asarray(self.base) + jnp.stack([(L * jnp.cos(ang)).sum(-1), (L * jnp.sin(ang)).sum(-1)], -1)

    def pose(self, q):
        return jnp.tile(jnp.asarray(self.base)[None], (q.shape[0], 1)), jnp.zeros(q.shape[0])

    def proprio(self, q, v):
        qq = q[:, : self.nq]
        return jnp.concatenate([jnp.cos(qq), jnp.sin(qq), v[:, : self.nq], self.effector(q)], -1)

    def track(self, q, v, target, heading):
        # Damped least-squares IK step on the tip position, then joint PD.
        qq = q[:, : self.nq]
        ang = self.angles(q)
        L = jnp.asarray(self.links)
        # d tip / d q_i = sum_{k >= i} L_k (-sin, cos)(angle_k)
        jx = jnp.flip(jnp.cumsum(jnp.flip(-L * jnp.sin(ang), -1), -1), -1)
        jy = jnp.flip(jnp.cumsum(jnp.flip(L * jnp.cos(ang), -1), -1), -1)
        J = jnp.stack([jx, jy], 1)  # (N, 2, nq)
        err = target - self.effector(q)
        err = err * jnp.minimum(1.0, 0.15 / (jnp.linalg.norm(err, axis=-1, keepdims=True) + 1e-6))
        JJt = J @ jnp.swapaxes(J, 1, 2) + 0.05 * jnp.eye(2)
        dq = (jnp.swapaxes(J, 1, 2) @ jnp.linalg.solve(JJt, err[..., None]))[..., 0]
        return jnp.clip(5.0 * dq - 1.0 * v[:, : self.nq], -1, 1)

    def init(self, key, n, centre, spread):
        import jax

        # folded near the base, tip low on the left: approaches blocks from below
        q0 = jnp.asarray([2.4] + [-2.2 if i == 1 else 1.0 for i in range(1, self.nq)])
        return q0 + 0.2 * jax.random.uniform(key, (n, self.nq), minval=-1, maxval=1)


EMBODIMENTS = {
    "point": Point("point", nq=2, act_dim=2, push_offset=0.06, oriented=False),
    "point_slow": Point("point_slow", nq=2, act_dim=2, push_offset=0.06, oriented=False, gear=0.8, time_scale=2.5),
    "point_heavy": Point("point_heavy", nq=2, act_dim=2, push_offset=0.06, oriented=False, mass=4.0, time_scale=1.5),
    "stick_short": Stick("stick_short", nq=3, act_dim=3, push_offset=0.025, half_length=0.08, turn_first=True, push_width=0.05, body_radius=0.11),
    "stick_medium": Stick("stick_medium", nq=3, act_dim=3, push_offset=0.025, half_length=0.15, turn_first=True, push_width=0.1, body_radius=0.18),
    "stick_long": Stick("stick_long", nq=3, act_dim=3, push_offset=0.025, half_length=0.3, turn_first=True, push_width=0.2, body_radius=0.33),
    "gripper": Gripper("gripper", nq=5, act_dim=4, push_offset=0.0, body_radius=0.12),
    "scoop": Scoop("scoop", nq=3, act_dim=3, push_offset=0.0, push_width=0.05, body_radius=0.16, turn_first=True),
    "diffdrive": DiffDrive("diffdrive", nq=3, act_dim=2, push_offset=0.0, push_width=0.05, body_radius=0.12),
    "arm2": Arm("arm2", nq=2, act_dim=2, push_offset=0.045, oriented=False, links=(0.8, 0.8)),
    "arm3": Arm("arm3", nq=3, act_dim=3, push_offset=0.045, oriented=False, links=(0.55, 0.55, 0.55)),
}
