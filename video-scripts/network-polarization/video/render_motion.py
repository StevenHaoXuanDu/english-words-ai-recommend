"""Global motion layers for render_shots.Show: the things that keep every frame alive.

* drift(t)        slow continuous camera drift (dolly +-0.8 %, float), applied to the whole
                  data plane at once - rects, brackets, axis ticks - so every rect keeps its
                  place on the axis (relative x never changes; at most ~8 px of global sway
                  at the row ends). The parallax dust and nebula carry most of the felt motion.
* life(sc, t)     ambient life in the data: +-1.5 px vertical breathing wave (y only), a soft
                  travelling glint (brightness + glow) and a tiny per-rect shimmer.
* round_pulse     on every round update a scanline of light passes down over the row
                  (all x at once: the update is synchronous) and briefly lifts the rects.
* trails          short afterimages behind rects that move in the world (camera moves
                  excluded): two faint smeared copies covering the last 0.07 s / 0.2 s of path.
* accents         ripple rings, spark bursts, radial glows for key moments.
* HUD model       live readouts (ROUND / eps / GROUPS / SPREAD) and sparkline data, all
                  computed from hk_sim states.

Only rect y, level and glow are touched here; x is never displaced by decoration.
"""
import math

import numpy as np

from render_core import RECT_W, Y0

TRAIL_DT = (0.07, 0.20)          # s: afterimage segment ends
TRAIL_LEVEL = (0.16, 0.06)       # ... and their brightness relative to the rect
TRAIL_MIN_PX = 4.0               # world px over TRAIL_DT[1] below which a rect leaves no trail
BREATH_PX, BREATH_LAMBDA, BREATH_T = 1.5, 900.0, 5.5
GLINT_PERIOD, GLINT_SIGMA = 7.0, 170.0
SHIMMER = 0.035


def clamp01(v):
    return min(1.0, max(0.0, float(v)))


def smooth(p):
    p = clamp01(p)
    return p * p * (3 - 2 * p)


def bump(u):
    return math.sin(math.pi * u) if 0.0 < u < 1.0 else 0.0


# ----------------------------------------------------------------------------- camera drift
def drift(t):
    """(zoom, dx, dy): a slow, never-repeating-looking dolly and float (periods 19-37 s)."""
    dz = 1.0 + 0.005 * math.sin(2 * math.pi * t / 23.0 + 0.7) + 0.003 * math.sin(2 * math.pi * t / 37.0 + 2.1)
    dx = 2.0 * math.sin(2 * math.pi * t / 29.0 + 0.3)
    dy = 4.0 * math.sin(2 * math.pi * t / 19.0 + 1.3) + 2.0 * math.sin(2 * math.pi * t / 31.0)
    return dz, dx, dy


def apply_drift(sc, d):
    dz, dx, dy = d
    fx_ = lambda x: (np.asarray(x) - 960.0) * dz + 960.0 + dx
    fy_ = lambda y: (np.asarray(y) - Y0) * dz + Y0 + dy
    for b in sc.rects:
        b["x"] = fx_(b["x"])
        b["y"] = fy_(b["y"])
        b["w"] = b["w"] * dz
        b["h"] = b["h"] * dz
        if b.get("fixed"):
            b["sx"], b["sy"] = b["sx"] * dz, b["sy"] * dz
    sc.lines = [(float(fx_(l[0])), float(fy_(l[1])), float(fx_(l[2])), float(fy_(l[3]))) + l[4:] for l in sc.lines]
    sc.dots = [(float(fx_(d_[0])), float(fy_(d_[1])), d_[2] * dz) + d_[3:] for d_ in sc.dots]
    for f in sc.fx:
        for k in ("x", "x0", "x1"):
            if k in f:
                f[k] = float(fx_(f[k]))
        if "y" in f:
            f["y"] = float(fy_(f["y"]))
        for k in ("r", "rx", "ry"):
            if k in f:
                f[k] = f[k] * dz
        if "xs" in f:
            f["xs"] = [float(fx_(x)) for x in f["xs"]]
        if "pts" in f:
            f["pts"] = [(float(fx_(p[0])), float(fy_(p[1])), p[2] * dz, p[3] * dz, p[4], p[5]) for p in f["pts"]]
    sc.zoom *= dz
    sc.drift = d


# ----------------------------------------------------------------------------- life
def life(sc, t, amount=1.0):
    if amount <= 1e-3:
        return
    xg = -300.0 + 2520.0 * ((t / GLINT_PERIOD) % 1.0)
    for b in sc.rects:
        if not b.get("life"):
            continue
        x = b["x"]
        n = len(x)
        z = b["w"] / RECT_W
        b["y"] = b["y"] + amount * BREATH_PX * z * np.sin(2 * np.pi * (x / BREATH_LAMBDA - t / BREATH_T))
        g = np.exp(-0.5 * ((x - xg) / GLINT_SIGMA) ** 2)
        i = np.arange(n)
        ph = ((i * 0.6180339887) % 1.0) * 2 * np.pi
        fr = 0.55 + 0.45 * ((i * 0.7548776662) % 1.0)
        sh = SHIMMER * np.sin(2 * np.pi * fr * t + ph)
        b["level"] = b["level"] * (1.0 + amount * (0.07 * g + sh))
        b["glow"] = b["glow"] * (1.0 + amount * 0.35 * g)


# ----------------------------------------------------------------------------- round pulse
def round_windows(ev):
    ks = sorted((v[0], v[1], int(k.split(":")[1])) for k, v in ev.items() if k.startswith("round:"))
    out = []
    for j, (a, b, k) in enumerate(ks):
        slot = (ks[j + 1][0] - a) if j + 1 < len(ks) else (b - a) * 1.4
        out.append((a, b, k, slot))
    return out


PULSE_LEN = 0.34


def round_pulse(sc, sh, t):
    """Scanline + lift on every round start, scaled down for very short rounds (S20) so a
    run of quick rounds never flickers. Returns the pulse strength (for the axis)."""
    best = 0.0
    for a, b, k, slot in round_windows(sh["events"]):
        amp = clamp01((slot - 0.25) / 0.5)
        u = (t - (a - 0.04)) / PULSE_LEN
        if amp <= 0 or not (0 < u < 1):
            continue
        main = [r for r in sc.rects if r.get("life")]
        if not main:
            continue
        ys = np.concatenate([r["y"][r["level"] > 0.02] for r in main] or [np.zeros(0)])
        xs = np.concatenate([r["x"][r["level"] > 0.02] for r in main] or [np.zeros(0)])
        if not len(ys):
            continue
        top, bot = float(ys.min()) - 40.0, float(ys.max()) + 40.0
        e = u * u * (3 - 2 * u)
        y = top + (bot - top) * e
        env = bump(u)
        for r in main:
            g = np.exp(-0.5 * ((r["y"] - y) / 18.0) ** 2) * env * amp
            r["level"] = r["level"] * (1 + 0.35 * g)
            r["glow"] = r["glow"] * (1 + 1.1 * g)
        # one scanline per occupied stretch of the row: never a line of light across a gap
        xs = np.sort(xs)
        cuts = np.flatnonzero(np.diff(xs) > 40.0)
        for lo, hi in zip(np.r_[0, cuts + 1], np.r_[cuts, len(xs) - 1]):
            sc.fx.append(dict(type="scan", y=y, x0=float(xs[lo]) - 10.0, x1=float(xs[hi]) + 10.0,
                              amp=0.13 * amp * env))
        best = max(best, amp * env)
    return best


# ----------------------------------------------------------------------------- trails
def _world(b, x, y):
    z, cx = b["cam"]
    return (x - 960.0) / z + cx, (y - Y0) / z + Y0


def trails(sc, past):
    """Ghost batch (always inserted first, so it lies under the rects) from the main batches
    of sc and of the same shot evaluated at t - TRAIL_DT (past = [scene_a, scene_b])."""
    xs, ys, ws, hs, cols, lvs, sxs, sys_ = [], [], [], [], [], [], [], []
    for bi, b in enumerate(sc.rects if past else []):
        if not b.get("life") or b.get("cam") is None:
            continue
        pa = [p.rects[bi] if bi < len(p.rects) else None for p in past]
        if any(p is None or len(p["x"]) != len(b["x"]) or p.get("cam") is None for p in pa):
            continue
        z, cx = b["cam"]
        w0x, w0y = _world(b, b["x"], b["y"])
        pw = [_world(p, p["x"], p["y"]) for p in pa]
        far = np.hypot(pw[1][0] - w0x, pw[1][1] - w0y)
        mv = (far > TRAIL_MIN_PX) & (b["level"] > 0.02)
        if not mv.any():
            continue
        # screen positions of the past world positions under the current camera
        P = [(b["x"], b["y"])] + [((wx - cx) * z + 960.0, (wy - Y0) * z + Y0) for wx, wy in pw]
        fade = np.clip((far - TRAIL_MIN_PX) / 8.0, 0.0, 1.0)
        for seg in range(2):
            (ax, ay), (bx, by) = P[seg], P[seg + 1]
            xs.append(0.5 * (ax + bx)[mv])
            ys.append(0.5 * (ay + by)[mv])
            sxs.append((ax - bx)[mv])
            sys_.append((ay - by)[mv])
            ws.append(b["w"][mv])
            hs.append(b["h"][mv])
            cols.append(b["rgb"][mv])
            lvs.append((np.minimum(b["level"], 1.0) * TRAIL_LEVEL[seg] * fade)[mv])
    if xs:
        sc.add_rects(np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(hs),
                     np.concatenate(cols), np.concatenate(lvs), 0.0,
                     smear=(np.concatenate(sxs), np.concatenate(sys_)))
    else:
        sc.add_rects(np.zeros(0), np.zeros(0), 0.0, 0.0, np.zeros((0, 3)), 0.0, 0.0, smear=(np.zeros(0), np.zeros(0)))
    sc.rects.insert(0, sc.rects.pop())


# ----------------------------------------------------------------------------- accents
def ripple(t, t0, x, y, rmax, dur=1.4, amp=0.24, ratio=0.28, w=2.4):
    u = (t - t0) / dur
    if not (0.0 < u < 1.0):
        return None
    e = 1 - (1 - u) ** 3
    return dict(type="ripple", x=x, y=y, r=8.0 + (rmax - 8.0) * e, ratio=ratio, amp=amp * (1 - u) ** 1.6, w=w)


def burst(t, t0, x, y, n=40, seed=0, speed=(160.0, 420.0), dur=1.3, amp=0.55, half=(60.0, 140.0),
          fps=30.0, squash=0.55):
    """Spark burst from an (x, y) +- half box. Sparks decelerate (tau 0.33 s), drift up
    slightly and fade; their per-frame displacement is the smear (no strobing)."""
    tau = t - t0
    if not (0.0 < tau < dur):
        return None
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    sp = rng.uniform(*speed, n)
    life_ = rng.uniform(0.55, 1.0, n) * dur
    ox = rng.uniform(-half[0], half[0], n)
    oy = rng.uniform(-half[1], half[1], n)
    size = rng.uniform(0.9, 1.8, n)
    td = 0.33
    pts = []
    for i in range(n):
        if tau >= life_[i]:
            continue
        dx, dy = math.cos(ang[i]), math.sin(ang[i]) * squash
        D = sp[i] * td * (1 - math.exp(-tau / td))
        v = sp[i] * math.exp(-tau / td)
        u = tau / life_[i]
        a = amp * (1 - u) ** 1.5 * min(1.0, tau / 0.05)
        px = x + ox[i] + dx * D
        py = y + oy[i] + dy * D - 18.0 * tau
        pts.append((px, py, dx * v / fps, (dy * v - 18.0) / fps, size[i], a))
    return dict(type="sparks", pts=pts)


def glow(t, t0, x, y, rx, ry, dur=0.9, amp=0.10, attack=0.12):
    u = t - t0
    if not (0.0 < u < dur):
        return None
    a = amp * min(1.0, u / attack) * (1 - smooth((u - attack) / (dur - attack)))
    return dict(type="glow", x=x, y=y, rx=rx, ry=ry, amp=a)


def beam(t, win, x_from=-160.0, x_to=2080.0, amp=0.16):
    t0, t1 = win
    if not (t0 < t < t1):
        return None, None
    u = (t - t0) / (t1 - t0)
    e = 0.7 * u + 0.3 * u * u * (3 - 2 * u)          # nearly constant speed across the row
    x = x_from + (x_to - x_from) * e
    env = min(1.0, u / 0.15) * min(1.0, (1 - u) / 0.15)
    return dict(type="beam", x=x, amp=amp * env), x
