"""README figures and animations, drawn with xmagix.draw (the same renderer as pixel observations).

    python scripts/gallery.py
writes docs/embodiments.png (every body on push), docs/tasks.png (every task, gripper) and
docs/<task>-<body>.gif (expert episodes).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import xmagix
from xmagix.draw import render

OUT = Path("docs")


def episode(task, emb, seed=3, frac=1.0):
    """Expert episode from one start: states (T, ...) and effector path (T, 2)."""
    env = xmagix.make(task, emb, 1, auto_reset=False)
    s = env.reset(jax.random.PRNGKey(seed))

    def body(s, k):
        s2, _, _ = env.step(s, env.expert(s), k)
        return s2, (s.q[0], s.v[0], env.effector(s.q)[0])

    _, (qs, vs, effs) = jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(seed + 1), int(env.horizon * frac)))
    return env, qs, vs, effs


def frame(env, q, v, trail, size=320):
    return np.asarray(render(env, q[None], v[None], size=size, supersample=2, trail=trail[None]))[0]


def sheet(tiles, cols, labels=None, pad=12, bg=(28, 30, 38)):
    rows = (len(tiles) + cols - 1) // cols
    H = tiles[0].shape[0]
    label_h = 26 if labels else 0
    out = Image.new("RGB", (cols * (H + pad) + pad, rows * (H + pad + label_h) + pad), bg)
    if labels:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(out)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = pad + c * (H + pad), pad + r * (H + pad + label_h)
        out.paste(Image.fromarray(t), (x, y + label_h))
        if labels:
            draw.text((x + 4, y + 6), labels[i], fill=(225, 228, 236))
    return out


def snapshot(task, emb, frac):
    env, qs, vs, effs = episode(task, emb, frac=frac)
    return frame(env, qs[-1], vs[-1], effs), env


def main():
    OUT.mkdir(exist_ok=True)
    tiles, labels = [], []
    for e in xmagix.EMBODIMENTS:
        img, env = snapshot("push", e, 0.3)
        tiles.append(img)
        labels.append(f"{e}  ·  {env.act_dim}-d action, {env.obs_dim}-d state")
    sheet(tiles, 4, labels).save(OUT / "embodiments.png")
    tiles, labels = [], []
    for t, f in [("reach", .6), ("push", .3), ("sweep", .5), ("match", .5), ("route_left", .45), ("route_right", .45),
                 ("orbit_ccw", .6), ("orbit_cw", .6)]:
        tiles.append(snapshot(t, "gripper", f)[0])
        labels.append(t)
    sheet(tiles, 4, labels).save(OUT / "tasks.png")
    for t, e in [("push", "gripper"), ("sweep", "stick_medium"), ("route_left", "point"), ("orbit_ccw", "arm2"),
                 ("match", "diffdrive")]:
        env, qs, vs, effs = episode(t, e)
        idx = list(range(0, len(qs), 2))
        frames = [Image.fromarray(frame(env, qs[i], vs[i], effs[: i + 1], size=240)) for i in idx]
        frames[0].save(OUT / f"{t}-{e}.gif", save_all=True, append_images=frames[1:], duration=50, loop=0,
                       optimize=True)
        print("wrote", OUT / f"{t}-{e}.gif", len(frames), "frames", flush=True)


if __name__ == "__main__":
    main()
