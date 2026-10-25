"""MJCF for one (task, embodiment, variant): a walled top-down arena, static obstacles, the
embodiment, and the blocks. Goal regions are drawn as non-colliding geoms (for rendering only).

Collision groups: walls/obstacles contype 2; blocks and mobile agents collide with both blocks
and walls; fixed-base arms collide with blocks only (they reach over walls).
"""

from xmagix.embodiments import Z, Embodiment
from xmagix.tasks import BLOCK, COLOURS, MatchRegions, PushToRegion, Reach, Route, Task

VARIANTS = {  # block physics
    "default": dict(mass=0.3, damping=2.0),
    "heavy": dict(mass=1.2, damping=6.0),  # blocks resist: slower, more effort-dependent pushing
    "slippery": dict(mass=0.3, damping=0.3),  # blocks keep sliding: momentum matters
}
STATIC = 'contype="2" conaffinity="0" rgba="0.35 0.35 0.4 1"'
REGION = 'contype="0" conaffinity="0" group="1"'


def build_xml(task: Task, emb: Embodiment, variant="default"):
    phys = VARIANTS[variant]
    walls = "".join(
        f'\n    <geom type="box" pos="{x} {y} {Z}" size="{hx} {hy} 0.04" {STATIC}/>'
        for x, y, hx, hy in [(0, 1.05, 1.1, 0.05), (0, -1.05, 1.1, 0.05), (1.05, 0, 0.05, 1.1), (-1.05, 0, 0.05, 1.1)])
    walls += "".join(f'\n    <geom type="box" pos="{x} {y} {Z}" size="{hx} {hy} 0.04" {STATIC}/>'
                     for x, y, hx, hy in task.obstacles)
    walls += "".join(f'\n    <geom type="cylinder" pos="{x} {y} {Z}" size="{r} 0.04" {STATIC}/>'
                     for x, y, r in task.pillars)
    regions = ""
    if isinstance(task, (PushToRegion, Route)):
        regions += f'\n    <geom type="cylinder" pos="{task.goal[0]} {task.goal[1]} 0.001" size="{task.radius} 0.001" rgba="0.2 0.7 0.3 0.35" {REGION}/>'
    if isinstance(task, MatchRegions):
        cx, cy, hx, hy = task.region
        regions += f'\n    <geom type="box" pos="{cx} {cy} 0.001" size="{hx} {hy} 0.001" rgba="0.2 0.7 0.3 0.35" {REGION}/>'
    if isinstance(task, Reach):
        regions += "".join(f'\n    <geom type="cylinder" pos="{x} {y} 0.001" size="{task.radius} 0.001" rgba="0.2 0.7 0.3 0.35" {REGION}/>'
                           for x, y in task.waypoints)
    if hasattr(task, "line"):
        regions += f'\n    <geom type="box" pos="0 {(task.line + 1) / 2} 0.001" size="1 {(1 - task.line) / 2} 0.001" rgba="0.2 0.7 0.3 0.25" {REGION}/>'
    body, actuators = emb.xml()
    blocks = "".join(f"""
    <body name="block{i}" pos="0 0 {Z}">
      <joint name="b{i}x" type="slide" axis="1 0 0" damping="{phys['damping']}"/>
      <joint name="b{i}y" type="slide" axis="0 1 0" damping="{phys['damping']}"/>
      <joint name="b{i}r" type="hinge" axis="0 0 1" damping="{0.02 * phys['damping']}"/>
      <geom type="box" size="{BLOCK} {BLOCK} 0.04" mass="{phys['mass']}" rgba="{COLOURS[c]}" contype="1" conaffinity="3"/>
    </body>""" for i, c in enumerate(task.blocks))
    return f"""<mujoco model="xmagix-{task.name}-{emb.name}-{variant}">
  <option timestep="0.01" gravity="0 0 0" integrator="implicitfast" iterations="20" ls_iterations="30"/>
  <default><geom friction="0.5 0.005 0.0001" condim="3" solref="0.01 1"/><joint armature="0.01"/></default>
  <worldbody>
    <geom type="plane" size="1.1 1.1 0.1" rgba="0.93 0.93 0.9 1" contype="0" conaffinity="0"/>{walls}{regions}{body}{blocks}
  </worldbody>
  <actuator>{actuators}
  </actuator>
</mujoco>"""
