"""Per-shot choreography for render.py.

Show(timeline).scene(t) -> render_core.Scene for absolute time t (seconds).
All times come from timeline.json (shot bounds + named events); all opinions and
positions come from hk_sim.run(). Nothing here hardcodes an absolute time.
Global motion (camera drift, ambient life, round pulses, trails) lives in render_motion,
the background in render_fx, the animated typography in render_text.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hk_sim  # noqa: E402

import render_motion as M  # noqa: E402
import timeline as TL  # noqa: E402
from render_core import (GOLD, RECT_H, RECT_W, WHITE_GOLD, Y0, ROW_L, ROW_R,  # noqa: E402
                         Layout, Scene, X, card_box, layout_seq, opinion_rgb)
from render_fx import AXIS_DY, AXIS_Y_PARALLAX  # noqa: E402

AXIS_CLEAR = 30.0     # design px (x zoom) the axis keeps below the lowest rect of its camera

N = hk_sim.N
X0 = np.asarray(hk_sim.X0, dtype=np.float64)


# ----------------------------------------------------------------------------- easing
def clamp01(v):
    return min(1.0, max(0.0, float(v)))


def ease(p):
    """easeInOutCubic"""
    p = clamp01(p)
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out(p):
    p = clamp01(p)
    return 1 - (1 - p) ** 3


def smooth(p):
    p = clamp01(p)
    return p * p * (3 - 2 * p)


def smooth_a(p):
    p = np.clip(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def ease_sine(p):
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(p))


def ease_out_expo(p):
    p = clamp01(p)
    return 1.0 if p >= 1.0 else 1 - 2 ** (-10 * p)


def cubic_bezier(x1, y1, x2, y2):
    """CSS-style cubic-bezier easing (endpoints 0 and 1); monotonic for y1, y2 in [0, 1]."""
    def b(a1, a2, u):
        return 3 * (1 - u) ** 2 * u * a1 + 3 * (1 - u) * u * u * a2 + u ** 3

    def f(p):
        p = clamp01(p)
        if p <= 0.0 or p >= 1.0:
            return p
        lo, hi = 0.0, 1.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if b(x1, x2, mid) < p:
                lo = mid
            else:
                hi = mid
        return b(y1, y2, 0.5 * (lo + hi))
    return f


# arrivals: a short, soft take-off and a long exponential-like landing; never overshoots
ease_snap = cubic_bezier(0.30, 0.0, 0.0, 1.0)
ease_land = cubic_bezier(0.25, 0.0, 0.15, 1.0)


def settle(p, head=True):
    """Bracket-edge snap weight for one eps animation window at progress p: 1 at rest,
    0 while the width moves (p in 0.2..0.8). head=False for brackets opening from zero."""
    if p <= 0.0:
        return 1.0 if head else 0.0
    if p >= 1.0:
        return 1.0
    w = smooth((p - 0.8) / 0.2)
    return max(w, 1 - smooth(p / 0.2)) if head else w


def bump(u):
    return math.sin(math.pi * u) if 0.0 < u < 1.0 else 0.0


def prog(t, win):
    t0, t1 = win
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return clamp01((t - t0) / (t1 - t0))


def window_fade(t, win, fade):
    """1 inside [t0, t1] with `fade`-second smooth ramps inside both ends."""
    t0, t1 = win
    f = min(fade, (t1 - t0) / 2)
    if f <= 0:
        return 0.0
    return smooth(min((t - t0) / f, (t1 - t) / f))


# ----------------------------------------------------------------------------- camera
class Cam:
    def __init__(self, z=1.0, cx=960.0):
        self.z, self.cx = float(z), float(cx)

    def x(self, wx):
        return (np.asarray(wx) - self.cx) * self.z + 960.0

    def y(self, wy):
        return (np.asarray(wy) - Y0) * self.z + Y0


CAM1 = Cam()


# ----------------------------------------------------------------------------- simulation
# Every state sequence is laid out round after round against the state before it (layout_seq),
# so blocks keep their slots, and consecutive tweens share the very same Layout object.
INIT = Layout(X0)                                                # the evenly spaced row
WIDE = layout_seq(hk_sim.run(X0, 0.30), INIT)                    # Experiment 1, rounds 0..4
_narrow = hk_sim.run(X0, 0.20)                                   # Experiment 2, rounds 0..7
NARROW = layout_seq(_narrow, INIT)
# S12 plays every round 1..7: in round 7 agents 49/50 make their last ~8 px move into the blocks.
NSEQ = NARROW
DEMO1 = NARROW[1]                                                # S07: run(X0, 0.20)[1]

BRIDGE_I = 49                     # top-right rect of the left 5x10 block
_eps_b = np.full(N, 0.20)
_eps_b[BRIDGE_I] = 0.40
BRIDGE = layout_seq(hk_sim.run(_narrow[-1], _eps_b), NARROW[-1])  # starts from S13's end state
MERGE_ROUND = next(k for k, L in enumerate(BRIDGE) if len(L.groups) == 1)
LEFT = np.arange(N) < BRIDGE_I    # left block without the bridge
RIGHT = np.arange(N) > BRIDGE_I

DEMO_I = 5
NB5 = np.where(np.abs(X0 - X0[DEMO_I]) <= 0.20 + 1e-12)[0]
MEAN5 = float(X0[NB5].mean())
DEMO_LIFT = 46.0                  # S06: agent 5 rises this high to pass over its neighbours
_x = X0.copy()
_x[DEMO_I] = MEAN5
_wy = INIT.wy.copy()
_wy[DEMO_I] -= DEMO_LIFT          # it lands from above: it takes the top of the 2-stack
MOVE5 = Layout(_x, prev_pos=(INIT.wx, _wy))
_MATE = [i for i in range(N) if i != DEMO_I and MOVE5.gsize[i] > 1 and abs(MOVE5.op[i] - MEAN5) < 1e-9]
DEMO_MATE = _MATE[0]              # agent 12: agent 5 lands on its opinion, they stack

CARD_S04 = "你只听和你差不多的人"


# ----------------------------------------------------------------------------- agents
class Ag:
    def __init__(self, op, wx, wy):
        self.op, self.wx, self.wy = op.copy(), wx.copy(), wy.copy()
        self.level = np.ones(N)
        self.glow = np.ones(N)
        self.hscale = None
        self.rgb = None

    @staticmethod
    def of(L):
        return Ag(L.op, L.wx, L.wy)

    @staticmethod
    def mix(A, B, e):
        return Ag(A.op + (B.op - A.op) * e, A.wx + (B.wx - A.wx) * e, A.wy + (B.wy - A.wy) * e)

    @staticmethod
    def tween(A, B, p):
        """Round tween at linear progress p: opinion/x/colour land over the first 68 % of the
        window (ease_snap: quick take-off, long soft landing, no overshoot), y (stacking) over
        the last half, so converging rects meet side by side and only then stack (a
        simultaneous x/y lerp draws diagonal '/' bars mid-tween)."""
        ex, ey = ease_snap(p / 0.68), ease_land((p - 0.5) / 0.5)
        return Ag(A.op + (B.op - A.op) * ex, A.wx + (B.wx - A.wx) * ex, A.wy + (B.wy - A.wy) * ey)

    def colours(self):
        if self.rgb is None:
            self.rgb = opinion_rgb(self.op)
        return self.rgb

    def emit(self, sc, cam, world_cam=None):
        """world_cam: the rects' wx/wy are already projected by world_cam (size follows it)."""
        z = (world_cam or cam).z
        h = RECT_H * z if self.hscale is None else RECT_H * z * self.hscale
        sc.add_rects(cam.x(self.wx), cam.y(self.wy), RECT_W * z, h,
                     self.colours(), self.level, self.glow, cam=((world_cam or cam).z, (world_cam or cam).cx),
                     life=True)


def round_pos(t, ev):
    """(k_from, k_to, linear progress) from the shot's round:k events."""
    ks = sorted(int(k.split(":")[1]) for k in ev if k.startswith("round:"))
    cur = ks[0] - 1
    for k in ks:
        t0, t1 = ev[f"round:{k}"]
        if t >= t1:
            cur = k
        elif t >= t0:
            return k - 1, k, clamp01((t - t0) / (t1 - t0))
        else:
            break
    return cur, cur, 0.0


def inside(op, xc, eps, soft=0.0004):
    """|x - xc| <= eps membership with a sub-pixel soft edge (0.0004 = 0.64 px), so the
    nearest non-members (0.001-0.003 beyond eps on the evenly spaced row) stay fully out."""
    return smooth_a((eps + soft - np.abs(op - xc)) / (2 * soft))


def _push(E, layout, side):
    """Keep a bracket edge from cutting through a stacked block: it jumps the block once
    the block's opinion is inside (side=+1 right edge, -1 left edge)."""
    if layout is None:
        return E
    for gx, hw, hh, g in layout.groups:
        if len(g) < 2:
            continue
        lim = hw + 12
        if side > 0:
            lo, hi = gx - lim, gx + 2
            if lo <= E <= hi:
                E += (gx + lim - E) * smooth((E - lo) / (hi - lo))
            elif hi < E < gx + lim:
                E = gx + lim
        else:
            lo, hi = gx - 2, gx + lim
            if lo <= E <= hi:
                E += (gx - lim - E) * smooth((hi - E) / (hi - lo))
            elif gx - lim < E < lo:
                E = gx - lim
    return E


EDGE_CLEAR = RECT_W / 2 + 2.5     # rect half width + line half width + 1.5 px air


def _snap(E, side, xc, eps, pos):
    """Move a bracket edge out of the single rect it would cut through, into the gap between
    the outermost member and the first non-member (membership itself is unchanged).
    side=+1 right edge, -1 left edge; pos = (op, wx, single) per agent."""
    op, wx, single = pos
    near = single & (np.abs(wx - E) < 2 * EDGE_CLEAR)
    if not near.any():
        return E
    d = op - xc
    member = np.abs(d) <= eps + 1e-12
    outside = (side * d > eps + 1e-12)
    lo, hi = -np.inf, np.inf
    if side > 0:
        if (near & member).any():
            lo = float(wx[near & member].max() + EDGE_CLEAR)
        if (near & outside).any():
            hi = float(wx[near & outside].min() - EDGE_CLEAR)
    else:
        if (near & outside).any():
            lo = float(wx[near & outside].max() + EDGE_CLEAR)
        if (near & member).any():
            hi = float(wx[near & member].min() - EDGE_CLEAR)
    if lo <= hi:
        return min(max(E, lo), hi)
    return 0.5 * (lo + hi)        # overlapping rects: no clean gap, split the difference


def bracket(sc, cam, xc, eps, opacity, layout=None, alpha=0.7, glow=0.35, half=None,
            snap=1.0, pos=None):
    """Bracket on opinion xc with half width eps. Edges jump past stacked blocks (_push) and,
    weighted by `snap` (1 = width at rest), sit in the gap next to single rects (_snap).
    pos = (op, wx) of the drawn rects when they differ from `layout`."""
    if opacity <= 1e-3:
        return
    L, R = _push(float(X(xc - eps)), layout, -1), _push(float(X(xc + eps)), layout, +1)
    if snap > 1e-3 and (layout is not None or pos is not None):
        op = layout.op if pos is None else pos[0]
        wx = layout.wx if pos is None else pos[1]
        single = layout.gsize == 1 if layout is not None else np.ones(len(op), bool)
        P = (np.asarray(op), np.asarray(wx), single)
        L += (_snap(L, -1, xc, eps, P) - L) * snap
        R += (_snap(R, +1, xc, eps, P) - R) * snap
    L = min(max(L, ROW_L), ROW_R)
    R = max(min(R, ROW_R), ROW_L)
    if half is None:
        half = 70.0
        if layout is not None:
            for gx, hw, hh, g in layout.groups:
                if gx + hw >= L and gx - hw <= R:
                    half = max(half, hh + 30.0)
    a = alpha * opacity
    y0, y1 = float(cam.y(Y0 - half)), float(cam.y(Y0 + half))
    tick = min(12.0 * cam.z ** 0.5, 0.3 * float(cam.x(R) - cam.x(L)))   # narrow: ticks don't meet
    for xe, d in ((L, 1.0), (R, -1.0)):
        sx = float(cam.x(xe))
        sc.line(sx, y0 - 1, sx, y1, 2.0, GOLD, a, glow)
        sc.line(sx, y0, sx + d * tick, y0, 2.0, GOLD, a, glow)


def threads(sc, cam, a, grow, alpha):
    """Thin gold lines from the facing edges of both blocks to the bridge rect."""
    if alpha <= 1e-3 or grow <= 0:
        return
    y = float(cam.y(a.wy[BRIDGE_I]))
    bx = a.wx[BRIDGE_I]
    lx, rx = a.wx[LEFT].max() + RECT_W / 2 + 3, a.wx[RIGHT].min() - RECT_W / 2 - 3
    bl, br = bx - RECT_W / 2 - 3, bx + RECT_W / 2 + 3
    if bl > lx:
        sc.line(float(cam.x(lx)), y, float(cam.x(lx + (bl - lx) * grow)), y, 1.0, GOLD, alpha, 0.5)
    if rx > br:
        sc.line(float(cam.x(rx)), y, float(cam.x(rx - (rx - br) * grow)), y, 1.0, GOLD, alpha, 0.5)


LISTEN_PERIOD, LISTEN_SIGMA = 1.6, 0.022
REF = {}      # absolute start times of effects that run across a cut (set by Show)


def listen(a, xc, eps, t, on, off=None, amp=1.0, period=None):
    """'Listening' wave: soft light runs out from the listener at opinion xc to the edges of
    its bracket, over the members only, every LISTEN_PERIOD s (level and glow only). It
    runs from `on`, fading in, and fades out before `off`."""
    if t <= on:
        return
    k = smooth((t - on) / 0.4) * (1.0 if off is None else 1 - smooth((t - off + 0.4) / 0.4))
    if k <= 1e-3:
        return
    u = ((t - on) / (period or LISTEN_PERIOD)) % 1.0
    d = np.abs(a.op - xc)
    g = np.exp(-0.5 * ((d - eps * u) / LISTEN_SIGMA) ** 2) * inside(a.op, xc, eps) * (1 - u) ** 0.6
    g *= k * amp
    a.level = a.level * (1 + 0.32 * g)
    a.glow = a.glow * (1 + 0.9 * g)


PULSE_EVERY, PULSE_TRAVEL = 0.55, 0.5


def thread_pulses(sc, cam, a, t, t_on, alpha):
    """Light pulses along the bridge threads, alternating: out from the bridge to both
    blocks, then in from both blocks to the bridge (it hears them, they hear it)."""
    if alpha <= 1e-3 or t <= t_on:
        return
    y = float(cam.y(a.wy[BRIDGE_I]))
    bx = a.wx[BRIDGE_I]
    lx, rx = a.wx[LEFT].max() + RECT_W / 2 + 3, a.wx[RIGHT].min() - RECT_W / 2 - 3
    bl, br = bx - RECT_W / 2 - 3, bx + RECT_W / 2 + 3
    xs, amps = [], []
    n = int((t - t_on) / PULSE_EVERY)
    for k in range(max(0, n - 1), n + 1):
        u = (t - t_on - k * PULSE_EVERY) / PULSE_TRAVEL
        if not (0 < u < 1):
            continue
        e = smooth(u)
        env = math.sin(math.pi * u) ** 0.7
        for a0, a1 in ((bl, lx), (br, rx)):        # outward on even pulses, inward on odd
            p0, p1 = (a0, a1) if k % 2 == 0 else (a1, a0)
            xs.append(float(cam.x(p0 + (p1 - p0) * e)))
            amps.append(0.42 * alpha * env)
    if xs:
        sc.fx.append(dict(type="pulses", y=y, xs=xs, amps=amps, sigma=4.5))


# ----------------------------------------------------------------------------- shots
AXIS_SEED = 0.05      # S01: the axis starts as a short hairline under the single rect


def s01(sh, t, sc):
    """One rect lights up; the background drifts in (its camera pulls back from 1.24 to 1, the
    dust depths part at their parallax and land at rest on the cut), a soft bloom opens
    around the rect and the axis is seeded."""
    E = sh["events"]
    sc.bg_level = smooth((t - sh["start"]) / 1.3)
    zb = 1.0 + 0.24 * (1 - ease_out(prog(t, (sh["start"], sh["end"]))))
    sc.cam = (zb, 960.0)
    sc.axis_cam = CAM1
    sc.axis_draw = AXIS_SEED * ease_land(prog(t, (E["rect_in"][0] + 0.35, sh["end"])))
    a = Ag.of(INIT)
    a.level[:] = 0.0
    a.level[50] = smooth(prog(t, E["rect_in"]))
    a.glow[50] = 1.0 + 0.22 * math.sin(2 * math.pi * (t - sh["start"]) / 3.0)
    a.emit(sc, CAM1)
    # a soft bloom of light opens around the first rect as it appears
    f = M.glow(t, E["rect_in"][0] + 0.25, float(INIT.wx[50]), Y0, 260.0, 150.0, dur=1.5, amp=0.06,
               attack=0.5)
    if f:
        sc.fx.append(f)


def s02(sh, t, sc):
    t0, t1 = sh["events"]["light_up"]
    fade = 0.4
    d = np.abs(np.arange(N) - 49.5)
    start = t0 + (d - 0.5) / 49.0 * max(t1 - t0 - fade, 1e-6)
    a = Ag.of(INIT)
    a.level = smooth_a((t - start) / fade)
    a.level[50] = 1.0
    a.emit(sc, CAM1)
    sc.axis_draw = AXIS_SEED + (1 - AXIS_SEED) * ease_land(prog(t, (t0 + 0.15, t1 + 0.2)))


S03_ZOOM, S03_RAMP, S03_FROM, S03_TO = 0.5, 0.4, 0.36, 0.64


def _s03_cam(sh, t):
    """Slow sweep: zoom 1 -> 1.6 and back with the centre panning X(0.36) -> X(0.64).
    Tuned so the median visible rect moves < 0.25 pitch per frame (no comb strobing)."""
    p = prog(t, sh["events"]["pan"])
    s = smooth(p / S03_RAMP) * (1 - smooth((p - 1 + S03_RAMP) / S03_RAMP))
    pan = X(S03_FROM) + (X(S03_TO) - X(S03_FROM)) * ease_sine(p)
    return Cam(1.0 + S03_ZOOM * s, 960.0 + (pan - 960.0) * s)


def _s03_flashes(sh):
    """8 deterministic (time, agent) pairs; each agent is distinct and inside the frame then."""
    rng = np.random.default_rng(1016)
    t0, t1 = sh["events"]["pan"]
    out, used = [], set()
    for k in range(8):
        tk = t0 + (t1 - t0) * (0.10 + 0.78 * (k + rng.uniform(0.15, 0.85)) / 8)
        cam = _s03_cam(sh, tk)
        lo = (cam.cx - 0.8 * 960 / cam.z - 160.0) / 1600.0    # visible opinions, 10 % margin
        hi = (cam.cx + 0.8 * 960 / cam.z - 160.0) / 1600.0
        cand = [i for i in range(N) if lo <= X0[i] <= hi and i not in used]
        i = int(cand[rng.integers(len(cand))])
        used.add(i)
        out.append((tk, i))
    return out


def s03(sh, t, sc):
    cam = _s03_cam(sh, t)
    a = Ag.of(INIT)
    for tk, i in _s03_flashes(sh):
        a.level[i] += 0.4 * bump((t - tk) / 0.3)
    a.emit(sc, cam)
    sc.zoom = cam.z
    sc.cam = (cam.z, cam.cx)


S04_PUSH = 0.05      # S04: a slow push in on the row under the card; S05's zoom starts from it


def s04_cam(sh, t):
    return Cam(1.0 + S04_PUSH * ease_sine(prog(t, (sh["start"], sh["end"]))), 960.0)


def s04(sh, t, sc):
    """The row dims to half under the card and keeps living: a wave of light runs out from the
    middle of the row to both ends (the rule, played on everyone) while the camera pushes in."""
    E = sh["events"]
    cam = s04_cam(sh, t)
    a = Ag.of(INIT)
    a.level[:] = 1 - 0.5 * smooth(prog(t, E["dim"])) + 0.5 * smooth(prog(t, E["undim"]))
    lvl = a.level[0]
    listen(a, X0[50], 0.52, t, E["card"][0] - 0.3, off=E["undim"][1], amp=0.8, period=1.9)
    a.emit(sc, cam)
    sc.axis_level = 0.55 + 0.45 * (lvl - 0.5) / 0.5
    sc.zoom = cam.z
    sc.cam = (cam.z, cam.cx)


S05_CAM = Cam(1.8, float(X(X0[DEMO_I])))   # centred on agent 5; both bracket edges stay in frame
DEMO_DIM = 0.15                  # S05-S07: the other rects sit this much lower than agent 5 ...
DEMO_GLOW = 0.5                  # ... and agent 5's glow is this much stronger
EPS_TIGHT = (RECT_W / 2 + 2.5) / 1600.0   # bracket hugging agent 5 alone, before it opens


def demo_focus(sh_s05, t):
    """0 -> 1 as the S05 zoom lands on agent 5 (it stays 1 through S06 until S07's reset)."""
    z1 = sh_s05["events"]["zoom_in"][1]
    return smooth(prog(t, (z1 - 0.3, z1 + 0.5)))


def demo_marker(sc, cam, wx, wy, o):
    """A small gold caret just above agent 5, following it: says which rect 「它」 is without
    touching its opinion colour."""
    if o <= 1e-3:
        return
    tip = wy - RECT_H / 2 - 8.0
    x, ty = float(cam.x(wx)), float(cam.y(tip))
    dx, dy = 6.0 * cam.z, 6.0 * cam.z
    sc.line(x - dx, ty - dy, x, ty, 2.0, GOLD, 0.9 * o, 0.6)
    sc.line(x + dx, ty - dy, x, ty, 2.0, GOLD, 0.9 * o, 0.6)


def focus_look(a, f, others=None):
    """Agent 5 at full brightness with a stronger glow; the others a touch lower."""
    a.level = a.level * (1 - DEMO_DIM * f) if others is None else others * (1 - DEMO_DIM * f)
    a.level[DEMO_I] = 1.0
    a.glow[DEMO_I] = 1.0 + DEMO_GLOW * f


def s05(sh, t, sc):
    E = sh["events"]
    e = ease(prog(t, E["zoom_in"]))           # a decisive push: the chapter transition
    z0 = 1.0 + S04_PUSH                       # where S04's slow push ended
    cam = Cam(z0 + (S05_CAM.z - z0) * e, 960 + (S05_CAM.cx - 960) * e)
    f = demo_focus(sh, t)
    # the bracket opens out of agent 5: it appears hugging agent 5 alone (everyone else dims),
    # then its edges sweep out to eps = 0.20 and the rects they pass light back up
    b0, b1 = E["bracket_in"]
    o = smooth(prog(t, (b0 - 0.8, b0 - 0.3)))
    pw = prog(t, (b0 - 0.3, b1 + 0.4))
    eps = EPS_TIGHT + (0.20 - EPS_TIGHT) * ease_sine(pw)
    a = Ag.of(INIT)
    ins = inside(a.op, X0[DEMO_I], eps, soft=0.0004 + 0.004 * (1 - smooth(pw)))
    focus_look(a, f, 1 - 0.8 * o * (1 - ins))
    listen(a, X0[DEMO_I], eps, t, b1 + 0.4)
    a.emit(sc, cam)
    bracket(sc, cam, X0[DEMO_I], eps, o, INIT, snap=settle(pw, head=False))
    demo_marker(sc, cam, a.wx[DEMO_I], a.wy[DEMO_I], f)
    sc.zoom = cam.z
    sc.cam = (cam.z, cam.cx)


POINT_RGB = 0.55 * opinion_rgb(MEAN5) + 0.45


MATE_SETTLE = 0.20     # agent 12 makes room in the 2-stack only in this last part of the move


def demo_move(p, back=False):
    """S06 move of agent 5 onto agent 12 at linear progress p (back=True: S07's reset, the same
    path played in reverse). Nobody else moves; agent 12 settles into the 2-stack only as
    agent 5 lands (and leaves it as soon as agent 5 takes off again)."""
    a = Ag.of(INIT)
    e = ease(p)
    u = 1 - e if back else e                   # 0 = start of the row, 1 = landed on agent 12
    i, m = DEMO_I, DEMO_MATE
    # rise clear of the neighbours, travel, then drop onto agent 12 (no clipping through rects)
    ux = smooth((u - 0.1) / 0.8)
    lift = DEMO_LIFT * smooth(u / 0.25) * (1 - smooth((u - 0.75) / 0.25))
    a.op[i] = INIT.op[i] + (MOVE5.op[i] - INIT.op[i]) * ux
    a.wx[i] = INIT.wx[i] + (MOVE5.wx[i] - INIT.wx[i]) * ux
    a.wy[i] = INIT.wy[i] + (MOVE5.wy[i] - INIT.wy[i]) * smooth(u) - lift
    s = 1 - smooth(p / MATE_SETTLE) if back else smooth((p - 1 + MATE_SETTLE) / MATE_SETTLE)
    a.wy[m] = INIT.wy[m] + (MOVE5.wy[m] - INIT.wy[m]) * s
    if abs(a.wx[i] - a.wx[m]) < RECT_W + 4.0:          # above agent 12: rest on it, a row up
        a.wy[i] = min(a.wy[i], a.wy[m] - (MOVE5.wy[m] - MOVE5.wy[i]))
    return a


def s06_follow(sh, t):
    """0 -> 1: after agent 5 lands, its bracket glides to its new opinion (its neighbourhood for
    the next round is centred on itself again)."""
    m1 = sh["events"]["move"][1]
    return prog(t, (m1 + 0.05, m1 + 0.95))


def s06(sh, t, sc):
    E = sh["events"]
    cam = S05_CAM
    m0, m1 = E["move"]
    a = demo_move(prog(t, E["move"]))
    pf = s06_follow(sh, t)
    xc = X0[DEMO_I] + (MEAN5 - X0[DEMO_I]) * ease(pf)
    ins = inside(a.op, xc, 0.20)
    focus_look(a, 1.0, 1 - 0.8 * (1 - ins))
    listen(a, X0[DEMO_I], 0.20, t, REF.get("S05_listen", sh["start"]), off=E["rays"][0] + 0.2)
    a.glow[[DEMO_I, DEMO_MATE]] *= 1 + 0.9 * bump((t - m1 + 0.05) / 0.5)      # landing
    listen(a, xc, 0.20, t, m1 + 0.35)
    a.emit(sc, cam)
    demo_marker(sc, cam, a.wx[DEMO_I], a.wy[DEMO_I], 1.0)
    bracket(sc, cam, xc, 0.20, 1.0, MOVE5 if t >= m1 else INIT, snap=settle(pf))
    r0, r1 = E["rays"]
    D = r1 - r0
    px, py = float(X(MEAN5)), Y0 - 110.0
    head, tail = ease((t - r0) / (0.6 * D)), ease((t - r0 - 0.4 * D) / (0.6 * D))
    if head > tail:
        for j in NB5:
            sx, sy = INIT.wx[j], INIT.wy[j] - RECT_H / 2 - 2
            ax, ay = sx + (px - sx) * tail, sy + (py - sy) * tail
            bx, by = sx + (px - sx) * head, sy + (py - sy) * head
            sc.line(float(cam.x(ax)), float(cam.y(ay)), float(cam.x(bx)), float(cam.y(by)),
                    1.2, GOLD, 0.55, 0.8)
    pi_ = smooth((t - r0 - 0.45 * D) / (0.4 * D)) * (1 - smooth((t - (m0 + m1) / 2) / ((m1 - m0) / 2)))
    sc.dot(float(cam.x(px)), float(cam.y(py)), 3.2 * cam.z, POINT_RGB, pi_, 3.0)
    sc.zoom = cam.z
    sc.cam = (cam.z, cam.cx)


def s07(sh, t, sc):
    E = sh["events"]
    ez = ease(prog(t, E["zoom_out"]))
    cam = Cam(S05_CAM.z + (1 - S05_CAM.z) * ez, S05_CAM.cx + (960 - S05_CAM.cx) * ez)
    pr = prog(t, E["reset_demo"])
    er = ease(pr)
    if t < E["round:1"][0]:
        a = demo_move(pr, back=True)
    elif t < E["rewind"][0]:
        a = _rounds(DEMO_SEQ, sh, t)
    else:
        a = REWIND_DEMO.at(ease(prog(t, E["rewind"])))
    xc = a.op[DEMO_I] if t < E["round:1"][0] else X0[DEMO_I]
    focus_look(a, 1 - er, 1 - 0.8 * (1 - er) * (1 - inside(X0, xc, 0.20)))
    a.emit(sc, cam)
    if t < E["round:1"][0]:
        demo_marker(sc, cam, a.wx[DEMO_I], a.wy[DEMO_I], 1 - smooth(pr / 0.7))
    bracket(sc, cam, xc, 0.20, 1 - er, MOVE5 if pr <= 0 else INIT, snap=settle(pr))
    pf = prog(t, E["flash"])
    if 0 < pf < 1:
        o = 0.12 * math.sin(math.pi * pf)
        for i in range(N):
            bracket(sc, cam, X0[i], 0.20, 1.0, INIT, alpha=o, glow=0.0,
                    half=46.0 + 34.0 * ((i * 0.6180339887) % 1.0))
    sc.zoom = cam.z
    sc.cam = (cam.z, cam.cx)


def sweep_light(sc, a, cam, t, win, amp=0.16, reverse=False):
    """Light-sweep beam across the frame; rects brighten as it passes (level/glow only).
    Returns the beam's screen x (None outside the window)."""
    f, bx = M.beam(t, win, *((2080.0, -160.0) if reverse else (-160.0, 2080.0)), amp=amp)
    if f is None:
        return None
    sc.fx.append(f)
    g = np.exp(-0.5 * ((cam.x(a.wx) - bx) / 70.0) ** 2) * (f["amp"] / amp)
    a.level = a.level * (1 + 0.25 * g)
    a.glow = a.glow * (1 + 1.0 * g)
    return bx


def s08(sh, t, sc):
    E = sh["events"]
    pin = prog(t, E["bracket_in"])
    ein, eout = ease(pin), ease(prog(t, E["bracket_out"]))
    a = Ag.of(INIT)
    a.level = 1 - 0.65 * ein * (1 - eout) * (1 - inside(a.op, X0[50], 0.30))
    listen(a, X0[50], 0.30, t, E["bracket_in"][1] + 0.1, off=E["bracket_out"][0] + 0.2)
    sweep_light(sc, a, CAM1, t, E["sweep"])
    a.emit(sc, CAM1)
    bracket(sc, CAM1, X0[50], 0.30 * ein, smooth(pin * 3) * (1 - eout), INIT,
            snap=settle(pin, head=False))


def _rounds(seq, sh, t):
    k0, k1, p = round_pos(t, sh["events"])
    return Ag.tween(seq[k0], seq[k1], p)


def s09(sh, t, sc):
    _rounds(WIDE, sh, t).emit(sc, CAM1)


def rise_glint(a, t, on, period=1.7, amp=1.0, sigma=26.0):
    """Light rising through a stacked block, bottom to top, every `period` s (level/glow only)."""
    if t <= on:
        return
    k = smooth((t - on) / 0.5) * amp
    u = ((t - on) / period) % 1.0
    top, bot = a.wy.min() - 40.0, a.wy.max() + 40.0
    yg = bot + (top - bot) * u
    g = np.exp(-0.5 * ((a.wy - yg) / sigma) ** 2) * k * math.sin(math.pi * u) ** 0.5
    a.level = a.level * (1 + 0.16 * g)
    a.glow = a.glow * (1 + 0.8 * g)


def s10(sh, t, sc):
    E = sh["events"]
    a = _rounds(WIDE, sh, t)
    rise_glint(a, t, E["accent"][0] + 0.6)
    a.emit(sc, CAM1)
    sc.glow_gain = (1.0 + 0.4 * bump(prog(t, E["glow"]))
                    + 0.06 * math.sin(2 * math.pi * (t - E["glow"][1]) / 2.6) * smooth((t - E["glow"][1]) / 0.6))
    t0 = E["accent"][0]
    xc = float(X(WIDE[-1].op.mean()))
    for f in (M.ripple(t, t0, xc, Y0 + 150.0, 820.0, dur=1.5, amp=0.22),
              M.burst(t, t0, xc, Y0, n=46, seed=10, half=(66.0, 150.0), amp=0.5),
              M.glow(t, t0, xc, Y0, 300.0, 230.0, dur=1.2, amp=0.075)):
        if f:
            sc.fx.append(f)


class Rewind:
    """A round sequence played backwards as one continuous motion. Each forward round moves x
    (and opinion) first and stacks (y) after, so backwards each round un-stacks first and then
    slides x back: the rects retrace the forward rounds and never cut diagonally across a block.
    Every one-axis move gets its own smoothstep window, as long as its largest displacement
    needs, and consecutive windows overlap a little, so the motion never stops dead between
    rounds (the speed only dips to ~40 % of its peak at each turn) while a column that is still
    un-stacking has barely started to slide sideways (no tilted '/' columns)."""

    OVERLAP = 0.15                 # fraction of the shorter of two neighbouring windows

    def __init__(self, seq):
        self.end = seq[-1]
        moves = []                 # (d_op, d_wx, d_wy, length)
        self.targets = []          # the round each move rewinds to
        for k in range(len(seq) - 1, 0, -1):
            A, B = seq[k], seq[k - 1]
            dy = B.wy - A.wy
            if np.abs(dy).max() > 0.5:
                moves.append((0.0, 0.0, dy, np.abs(dy).max()))
                self.targets.append(k - 1)
            dx = B.wx - A.wx
            if np.abs(dx).max() > 0.5 or np.abs(B.op - A.op).max() > 1e-9:
                moves.append((B.op - A.op, dx, 0.0, np.abs(dx).max()))
                self.targets.append(k - 1)
        self.n = len(seq) - 1
        lens = [max(m[3], 20.0) ** 0.6 for m in moves]
        wins, t = [], 0.0
        for k, L in enumerate(lens):
            if k:
                t -= self.OVERLAP * min(L, lens[k - 1])
            wins.append((t, t + L))
            t += L
        self.moves = [(m[0], m[1], m[2], (a / t, b / t)) for m, (a, b) in zip(moves, wins)]

    def round_at(self, u):
        """The round the rewind is heading to at progress u (len(seq) - 1 before it starts)."""
        r = self.n
        for (_, _, _, win), k in zip(self.moves, self.targets):
            if u > win[0]:
                r = k
        return r

    def at(self, u):
        """u = 0: the last state of seq; u = 1: back at its first state."""
        op, wx, wy = self.end.op.copy(), self.end.wx.copy(), self.end.wy.copy()
        for dop, dx, dy, win in self.moves:
            e = smooth(prog(u, win))
            op += dop * e
            wx += dx * e
            wy += dy * e
        return Ag(op, wx, wy)


REWIND = Rewind(WIDE)               # S11: the consensus block un-builds back to the spectrum
DEMO_SEQ = NARROW[:3]               # S07: rounds 1-2 of run(X0, 0.20) as the demo ...
REWIND_DEMO = Rewind(DEMO_SEQ)      # ... then played back to the start


def s11(sh, t, sc):
    E = sh["events"]
    a = REWIND.at(ease_sine(prog(t, E["rewind"])))
    pin = prog(t, E["bracket_in"])
    ein = ease(pin)
    pn = prog(t, E["narrow"])
    eps = 0.30 - 0.10 * ease(pn)
    a.level = 1 - 0.65 * ein * (1 - inside(a.op, X0[50], eps))
    listen(a, X0[50], eps, t, E["bracket_in"][1] + 0.15)
    a.emit(sc, CAM1)
    bracket(sc, CAM1, X0[50], eps * ein, smooth(pin * 3), INIT,
            snap=settle(pin, head=False) * settle(pn))


def s12(sh, t, sc):
    E = sh["events"]
    eo = ease(prog(t, E["bracket_out"]))
    a = _rounds(NSEQ, sh, t)
    a.level = 1 - 0.65 * (1 - eo) * (1 - inside(a.op, a.op[50], 0.20))
    a.emit(sc, CAM1)
    bracket(sc, CAM1, a.op[50], 0.20, 1 - eo, INIT)


GAP_13 = float(NSEQ[-1].op.max() - NSEQ[-1].op.min())     # 0.387: the gap between the clusters


def s13(sh, t, sc):
    """The split locks: a brief cool dip runs through both blocks (no ripple: a lock, not a
    celebration) and a hairline gap-measure snaps out between the two cluster centres, labelled
    with the gap from hk_sim."""
    E = sh["events"]
    a = Ag.of(NSEQ[-1])
    t0 = E["lock"][0]
    u = t - t0
    dip = 0.22 * min(1.0, u / 0.12) * (1 - smooth((u - 0.12) / 0.9)) if u > 0 else 0.0
    a.level = a.level * (1 - dip)
    a.glow = a.glow * (1 - 1.6 * dip)
    a.emit(sc, CAM1)
    sc.glow_gain = 1.0 + 0.04 * math.sin(2 * math.pi * (t - sh["start"]) / 3.0)
    gap_measure(sc, t, E["measure"], NSEQ[-1])


def gap_measure(sc, t, win, L, y=None):
    """Hairline between the leftmost and rightmost cluster centres of layout L, snapping out from
    the middle with end ticks, the gap (from the state's opinions) typed above it."""
    m0, m1 = win
    p = prog(t, (m0, m0 + 0.45))
    o = smooth((t - m0) / 0.12) * (1 - smooth((t - m1 + 0.35) / 0.35))
    if o <= 1e-3:
        return
    xl, xr = float(X(L.op.min())), float(X(L.op.max()))
    top = float(L.wy.min()) - RECT_H / 2
    y = (top - 30.0) if y is None else y
    e = ease_out_expo(p)
    xm = 0.5 * (xl + xr)
    a, b = xm + (xl - xm) * e, xm + (xr - xm) * e
    sc.line(a, y, b, y, 1.0, GOLD, 0.55 * o, 0.45)
    tk = smooth((p - 0.7) / 0.3)
    if tk > 1e-3:
        for xe in (xl, xr):
            sc.line(xe, y - 7.0 * tk, xe, y + 7.0 * tk, 1.0, GOLD, 0.6 * o, 0.45)
    lab = smooth((t - m0 - 0.35) / 0.3) * o
    if lab > 1e-3:
        sc.labels.append((f"{L.op.max() - L.op.min():.2f}", xm, y - 16.0, 0.62 * lab, 14))


S14_PUSH = 0.3


def s14(sh, t, sc):
    cam = Cam(1.0 + S14_PUSH * ease(prog(t, sh["events"]["push"])), float(X(0.5)))
    Ag.of(NSEQ[-1]).emit(sc, cam)
    sc.zoom = cam.z
    sc.cam = (cam.z, cam.cx)


S15_FAR = 3.2         # the split zooms to this times its S14 framing before it is gone


def s15(sh, t, sc):
    """Push-through: the S14 push accelerates into the gap until the split rushes past the
    edges (motion-blurred), and the spectrum lands from a pull-back. The dust follows one
    continuous camera (in, then back out to 1), so the background never jumps."""
    E = sh["events"]
    r0, r1 = E["reset"]
    u = prog(t, (r0, r1))
    z14 = 1.0 + S14_PUSH
    pa = clamp01(u / 0.36)
    za = z14 * (1 + (S15_FAR - 1) * pa ** 2.4)
    old = Ag.of(NSEQ[-1])
    old.level[:] = 1 - smooth((pa - 0.55) / 0.45)
    cam_old = Cam(za, float(X(0.5)))
    old.emit(sc, cam_old)
    pb = clamp01((u - 0.30) / 0.70)
    zb = 1.0 + 0.42 * (1 - ease_out_expo(pb))
    new = Ag.of(INIT)
    new.level[:] = smooth(pb / 0.3)
    cam_new = Cam(zb, 960.0)
    pin = prog(t, E["bracket_in"])
    eps = s15_eps(sh, t)
    listen(new, X0[50], eps, t, E["bracket_in"][1] + 0.15)
    new.emit(sc, cam_new)
    w = smooth((u - 0.30) / 0.08)
    sc.zoom = za * (1 - w) + zb * w
    # one continuous camera for the background: in to za, then back out to 1
    zbg = za if u < 0.36 else 1.0 + (z14 * S15_FAR - 1.0) * (1 - ease_out_expo((u - 0.36) / 0.64))
    sc.cam = (zbg, 960.0)
    sc.axis_cam = cam_new if u >= 0.30 else cam_old
    sc.axis_level = (1 - smooth(pa / 0.5)) if u < 0.30 else smooth((pb - 0.2) / 0.5)
    f = M.glow(t, r0 + 0.30 * (r1 - r0), 960.0, Y0, 520.0, 260.0, dur=0.6, amp=0.06, attack=0.1)
    if f:
        sc.fx.append(f)
    bracket(sc, CAM1, X0[50], eps, smooth(pin * 3), INIT, snap=settle(pin, head=False) * settle(s15_quiet(sh, t)))


def s15_quiet(sh, t):
    """Progress of the bracket's quiet narrowing under 「但有很多东西，在悄悄收窄括号。」"""
    c = sh["captions"][-1]
    return prog(t, (c["start"] + 0.3, c["end"]))


def s15_eps(sh, t):
    return 0.30 * ease(prog(t, sh["events"]["bracket_in"])) - (0.30 - S16_EPS[0]) * ease_sine(s15_quiet(sh, t))


S16_EPS = [0.275, 0.25, 0.20, 0.15, 0.10]   # illustrative (no values on screen); S15 ends at [0]
S16_DIM = [1.0, 0.70, 0.50, 0.35, 0.25]
_rng = np.random.default_rng(316)
_far = (np.abs(X0 - X0[50]) > S16_EPS[3] + 0.02) & (np.arange(N) >= 4) & (np.arange(N) <= N - 5)
S16_BLOCKED = np.sort(np.concatenate([
    _rng.choice(np.where(_far & (X0 < X0[50]))[0], 4, replace=False),
    _rng.choice(np.where(_far & (X0 > X0[50]))[0], 4, replace=False)]))
_d50 = X0 - X0[50]
_drift_mask = (np.abs(_d50) <= S16_EPS[4] + 1e-9) & (np.arange(N) != 50)
# up to 10 px next to the centre, tapering linearly to 0 at the bracket edge (neighbours close
# up by only ~1 px, and the outermost members keep their gap to the first non-member)
S16_DRIFT = np.where(_drift_mask, np.sign(_d50) * 10.0 * (1 - np.abs(_d50) / S16_EPS[4]), 0.0)


def _s16_state(e, p3=None):
    """(eps, level per agent, x offset per agent, height scale) after beat progress e[0..3].
    p3: linear progress of beat 3 (blocked rects flash, collapse to a line, go dark)."""
    eps = S16_EPS[0] + sum((S16_EPS[k] - S16_EPS[k - 1]) * e[k - 1] for k in range(1, 5))
    dim = S16_DIM[0] + sum((S16_DIM[k] - S16_DIM[k - 1]) * e[k - 1] for k in range(1, 5))
    ins = inside(X0, X0[50], eps)
    level = ins + (1 - ins) * dim
    hs = np.ones(N)
    if p3 is None:
        level[S16_BLOCKED] *= 1 - e[2]
    else:
        level[S16_BLOCKED] *= (1 + 0.25 * bump(p3 / 0.3)) * (1 - smooth((p3 - 0.3) / 0.6))
        hs[S16_BLOCKED] = 1 - 0.93 * ease_land((p3 - 0.15) / 0.55)
    return eps, level, S16_DRIFT * e[3], hs


def s16_flash(p3):
    """Glow lift of the blocked rects as they switch off (beat 3)."""
    return 1 + 1.6 * bump(p3 / 0.3)


def s16_beats(sh, t):
    """Per beat: (narrowing progress, bracket snap weight). Each beat narrows quickly at first
    (70 % in 0.7 s) and then keeps creeping until just before the next beat, so the bracket and
    the dimming of everyone outside it never stop; the edges settle into a gap only at the end."""
    E = sh["events"]
    starts = [E[f"beat:{k}"][0] for k in range(1, 5)]
    ends = starts[1:] + [sh["end"] - 0.1]
    es, snaps = [], []
    for a0, b0 in zip(starts, ends):
        D = max(b0 - 0.2 - a0, 0.8)
        u = t - a0
        es.append(clamp01(0.7 * ease_out(u / 0.7) + 0.3 * smooth(u / D)) if u > 0 else 0.0)
        snaps.append(1.0 if u <= 0 else max(1 - smooth(u / 0.12), smooth((u - D + 0.3) / 0.3)))
    return es, snaps


def s16(sh, t, sc):
    E = sh["events"]
    es, snaps = s16_beats(sh, t)
    b3 = E["beat:3"]
    p3 = prog(t, (b3[0], b3[0] + 0.75))
    eps, level, dx, hs = _s16_state(es, p3)
    a = Ag.of(INIT)
    a.level, a.wx = level, a.wx + dx
    a.hscale = hs
    a.glow[S16_BLOCKED] *= s16_flash(p3)
    listen(a, X0[50], eps, t, REF.get("S15_listen", sh["start"]))
    a.emit(sc, CAM1)
    for j, i in enumerate(S16_BLOCKED):
        f = M.burst(t, b3[0] + 0.28, float(INIT.wx[i]), Y0, n=7, seed=160 + j, speed=(60.0, 160.0),
                    dur=0.8, amp=0.45, half=(2.0, 3.0), squash=0.8)
        if f:
            sc.fx.append(f)
    bracket(sc, CAM1, X0[50], eps, 1.0, INIT, snap=math.prod(snaps), pos=(INIT.op, a.wx))


def s17(sh, t, sc):
    E = sh["events"]
    er = ease(prog(t, E["restore"]))
    eps, level, dx, _ = _s16_state([1.0] * 4)
    a = _rounds(NSEQ, sh, t)
    a.level = level + (1 - level) * er
    a.wx = a.wx + dx * (1 - er)
    tf = E["freeze"][0]
    cam = s17_cam(sh, t)
    if t >= tf:
        u = (t - tf) / 0.15
        f = bump(u)
        sc.glow_gain = 1.0 - 0.3 * smooth(u) + 0.9 * f
        a.level = a.level * (1 + 0.25 * f)
    a.emit(sc, cam)
    bracket(sc, CAM1, X0[50], eps, 1 - er, INIT, pos=(INIT.op, a.wx))
    sc.cam = (cam.z, cam.cx)
    sc.zoom = cam.z


S17_PUSH = 0.06       # the freeze: a slow push in on the frozen fracture


def s17_cam(sh, t):
    tf = sh["events"]["freeze"][0]
    return Cam(1.0 + S17_PUSH * ease_sine(prog(t, (tf, sh["end"]))), 960.0)


def _bridge_look(a, amount=1.0):
    rgb = a.colours().copy()
    rgb[BRIDGE_I] = rgb[BRIDGE_I] + (WHITE_GOLD - rgb[BRIDGE_I]) * amount
    a.rgb = rgb
    a.glow[BRIDGE_I] = 1.0 + 0.6 * amount


def s18(sh, t, sc):
    """Light-sweep wipe: the beam crosses left to right; behind it S17's frozen fracture is
    gone and the settled split (S13's end state) is back. Then the bridge lights up."""
    E = sh["events"]
    old = Ag.of(NSEQ[3])
    new = Ag.of(BRIDGE[0])
    cam_old = Cam(1.0 + S17_PUSH, 960.0)          # where S17's push ended
    old.wx = cam_old.x(old.wx)                     # (old rects drawn in screen space below)
    old.wy = cam_old.y(old.wy)
    zw = 1.0 + S17_PUSH * (1 - ease(prog(t, E["reset"])))
    sc.cam = (zw, 960.0)                           # dust settles back with the wipe
    f, bx = M.beam(t, E["reset"], amp=0.17)
    # the axis under each rect set uses that set's camera, split at the beam
    sc.axis_cam = CAM1 if t >= E["reset"][1] else cam_old
    if bx is not None:
        sc.axis_split = (bx, CAM1, cam_old)
    if bx is None:
        done = t >= E["reset"][1]
        old.level[:] = 0.0 if done else 1.0
        new.level[:] = 1.0 if done else 0.0
    else:
        sc.fx.append(f)
        old.level = smooth_a((old.wx - bx) / 50.0 + 0.5)
        new.level = smooth_a((bx - new.wx) / 50.0 + 0.5)
        g = np.exp(-0.5 * ((new.wx - bx) / 70.0) ** 2)
        new.level = new.level * (1 + 0.2 * g)
        new.glow = new.glow * (1 + 0.9 * g)
    old.emit(sc, Cam(1.0, 960.0), world_cam=cam_old)
    _bridge_look(new, ease(prog(t, E["highlight"])))
    pin = prog(t, E["bracket_in"])
    eps = 0.20 * ease(pin) + 0.20 * ease(prog(t, E["widen"]))
    listen(new, BRIDGE[0].op[BRIDGE_I], eps, t, E["bracket_in"][1])
    new.emit(sc, CAM1)
    bracket(sc, CAM1, BRIDGE[0].op[BRIDGE_I], eps, smooth(pin * 3), BRIDGE[0],
            snap=settle(pin, head=False) * settle(prog(t, E["widen"])))


def s19(sh, t, sc):
    E = sh["events"]
    k0, k1, e = round_pos(t, E)
    a = Ag.tween(BRIDGE[k0], BRIDGE[k1], e)
    _bridge_look(a)
    a.emit(sc, CAM1)
    r0, r1 = E["round:1"]
    bracket(sc, CAM1, a.op[BRIDGE_I], 0.40, 1 - smooth(prog(t, (r1 - 0.5, r1 + 0.4))),
            BRIDGE[k1] if e > 0.5 else BRIDGE[k0])
    threads(sc, CAM1, a, ease(prog(t, E["threads"])), 0.5)
    thread_pulses(sc, CAM1, a, t, E["threads"][1] - 0.1, 1.0)


def _contact_frac():
    """Progress of the merge tween (Ag.tween, BRIDGE round 35 -> 36) at which the two blocks
    touch: the facing columns are at most one rect width apart (edge gap <= RECT_W)."""
    A, B = BRIDGE[MERGE_ROUND - 1], BRIDGE[MERGE_ROUND]
    for p in np.linspace(0.0, 1.0, 4001):
        a = Ag.tween(A, B, p)
        if a.wx[RIGHT].min() - a.wx[LEFT].max() <= 2 * RECT_W:
            return float(p)
    return 1.0


MERGE_CONTACT = _contact_frac()


def merge_time(sh):
    """Absolute time the blocks touch in S20 (HUD switch, accent, glow)."""
    a, b = sh["events"][f"round:{MERGE_ROUND}"]
    return a + MERGE_CONTACT * (b - a)


def creep_index(sh, t):
    """Fractional round index of S20's creep at t (rounds 2..35 over the 'creep' event), from
    the same speed profile timeline.creep used to place the round windows: integer crossings
    land exactly on the window edges. None outside the creep."""
    E = sh["events"]
    if "creep" not in E:
        return None
    t0, t1 = E["creep"]
    if t < t0 or t > t1:
        return None
    ks = sorted(int(k.split(":")[1]) for k in E if k.startswith("round:"))
    ks = [k for k in ks if k < MERGE_ROUND]
    n = ks[-1] - ks[0] + 1
    return ks[0] - 1 + n * float(TL.creep_profile((t - t0) / (t1 - t0)))


def s20(sh, t, sc):
    """Rounds 2-35 are one continuous creep: one fractional round index (creep_index) runs
    through the states, interpolated linearly between consecutive ones (every state is passed
    through exactly, no per-round easing or hold, so nothing judders). The merge round is a
    normal tween."""
    E = sh["events"]
    k0, k1, e = round_pos(t, E)
    ci = creep_index(sh, t)
    if ci is not None:
        k = min(int(math.floor(ci)), MERGE_ROUND - 2)
        a = Ag.mix(BRIDGE[k], BRIDGE[k + 1], clamp01(ci - k))
    elif k1 < MERGE_ROUND:
        a = Ag.of(BRIDGE[k1])
    else:
        a = Ag.tween(BRIDGE[k0], BRIDGE[k1], e)
    _bridge_look(a)
    a.emit(sc, CAM1)
    tm = merge_time(sh)
    r0 = E[f"round:{MERGE_ROUND}"][0]
    th = 1 - smooth(prog(t, (r0, tm)))
    threads(sc, CAM1, a, 1.0, 0.5 * th)
    thread_pulses(sc, CAM1, a, t, REF.get("S19_pulses", sh["start"]), th)
    xc = float(X(BRIDGE[MERGE_ROUND].op.mean()))
    sc.glow_gain = 1.0 + 0.35 * bump((t - tm + 0.05) / 1.0)
    for f in (M.ripple(t, tm, xc, Y0 + 150.0, 980.0, dur=1.7, amp=0.25),
              M.burst(t, tm, xc, Y0, n=60, seed=20, half=(66.0, 150.0), amp=0.55,
                      speed=(180.0, 480.0), dur=1.5),
              M.glow(t, tm, xc, Y0, 340.0, 250.0, dur=1.4, amp=0.085)):
        if f:
            sc.fx.append(f)


DIAGRAM_SCALE, DIAGRAM_CY = 0.45, 245.0      # S21 schematic: scaled copy of the bridge-round-1 layout


def _diagram(sc, t, E):
    o = smooth(prog(t, E["diagram_in"])) * (1 - smooth(prog(t, E["diagram_out"])))
    if o <= 1e-3:
        return
    eps_o = 0.20                                   # everyone but the bridge
    base = BRIDGE[1]                               # bridge at 0.50, blocks at 0.31 / 0.69
    d0 = base.op[BRIDGE_I] - base.op[0]            # bridge-to-block distance (0.19)
    b0, b1 = E["break"]
    i1 = E["diagram_in"][1]
    pa = prog(t, (i1 + 0.2, b0))
    d = d0 + (eps_o - d0) * pa * pa                # creep up to the limit eps, then past it
    d += (0.27 - eps_o) * ease_out(prog(t, (b0, b1 + 0.6)))
    k = DIAGRAM_SCALE
    shift = (d - d0) * 1600.0 * k
    xs = (base.wx - X(base.op[BRIDGE_I])) * k + 960.0
    ys = (base.wy - Y0) * k + DIAGRAM_CY
    xs = np.where(LEFT, xs - shift, np.where(RIGHT, xs + shift, xs))
    lv = np.full(N, 0.9 * o)
    gl = np.ones(N)
    rgb = opinion_rgb(base.op)
    rgb[BRIDGE_I] = WHITE_GOLD
    lv[BRIDGE_I], gl[BRIDGE_I] = o, 1.6
    tension = smooth(pa) if t < b0 else 0.0        # the threads tighten as the gap nears 2 eps
    gl[BRIDGE_I] *= 1 + 0.5 * tension
    sc.add_rects(xs, ys, RECT_W * k, RECT_H * k, rgb, lv, gl, life=True)
    gap = 3.0
    bx, cy = xs[BRIDGE_I], ys[BRIDGE_I]
    pulses_x, pulses_a = [], []
    for side, (a_x, b_x) in ((-1, (xs[LEFT].max() + RECT_W * k / 2 + gap, bx - RECT_W * k / 2 - gap)),
                             (1, (xs[RIGHT].min() - RECT_W * k / 2 - gap, bx + RECT_W * k / 2 + gap))):
        if t < b0:
            sc.line(a_x, cy, b_x, cy, 1.0, GOLD, (0.5 + 0.3 * tension) * o, 0.5 + 0.4 * tension)
            # light pulses along the thread, alternating out / in (as in S19), until it snaps
            n = int((t - i1) / 0.5)
            for j in range(max(0, n - 1), n + 1):
                u = (t - i1 - j * 0.5) / 0.45
                if 0 < u < 1:
                    p0, p1 = (b_x, a_x) if j % 2 == 0 else (a_x, b_x)
                    pulses_x.append(p0 + (p1 - p0) * smooth(u))
                    pulses_a.append(0.38 * o * math.sin(math.pi * u) ** 0.7)
        else:                                      # snap: halves recoil to their ends and fade
            u = ease_out(prog(t, (b0, b1 + 0.15)))
            mid = (a_x + b_x) / 2
            al = 0.6 * o * (1 - smooth(prog(t, (b0, b1 + 0.25))))
            sc.line(a_x, cy, a_x + (mid - a_x) * (1 - u), cy, 1.0, GOLD, al, 0.5)
            sc.line(b_x, cy, b_x + (mid - b_x) * (1 - u), cy, 1.0, GOLD, al, 0.5)
    if pulses_x:
        sc.fx.append(dict(type="pulses", y=cy, xs=pulses_x, amps=pulses_a, sigma=3.0))
    # reach measure under the diagram: bridge-to-cluster distance lines stretch outward; the
    # fixed ticks mark eps = 0.20 (everyone but the bridge); the threads snap as they pass it
    ym = ys.min() - RECT_H * k / 2 - 16.0          # above the schematic, clear of the row below
    reach = eps_o * 1600.0 * k
    om = o * (1 - smooth(prog(t, (b1, b1 + 0.6))))
    if om > 1e-3:
        for side, xc_ in ((-1, xs[LEFT].mean()), (1, xs[RIGHT].mean())):
            sc.line(bx, ym, xc_, ym, 1.0, GOLD, 0.40 * om, 0.3)
            sc.line(xc_, ym - 4.0, xc_, ym + 4.0, 1.0, GOLD, 0.45 * om, 0.3)
            xr = bx + side * reach
            sc.line(xr, ym - 7.0, xr, ym + 7.0, 1.0, GOLD, (0.5 + 0.3 * tension) * om, 0.4)
            sc.labels.append(("ε", xr, ym - 13.0, 0.5 * om, 13))
        sc.line(bx, ym - 5.0, bx, ym + 5.0, 1.0, GOLD, 0.45 * om, 0.3)


def s21(sh, t, sc):
    E = sh["events"]
    a = Ag.of(BRIDGE[MERGE_ROUND])
    _bridge_look(a)
    a.level[:] = 1 - 0.6 * smooth(prog(t, E["dim"])) + 0.6 * smooth(prog(t, E["undim"]))
    a.emit(sc, CAM1)
    sc.axis_level = a.level[0]
    _diagram(sc, t, E)


CARD_MIN_LEAD = 0.35     # s: shortest time the rects under the S22 card get to go dark


def s22_schedule(sh, text):
    """Collapse stagger (fade start per agent, fade length per agent) and the card window.
    Rects whose box meets the question card's ink box (+8 px) fade first and are dark by
    card.t0, so the question is never drawn over lit rects; if the lead tightens the gap
    between collapse.t0 and card.t0 below CARD_MIN_LEAD, the card waits instead."""
    E = sh["events"]
    c0, c1 = E["collapse"]
    k0, k1 = E["card"]
    D = c1 - c0
    L = BRIDGE[MERGE_ROUND]
    order = np.random.default_rng(22).random(N)
    start, dur = c0 + order * 0.55 * D, np.full(N, 0.45 * D)
    if text is None:
        return start, dur, (k0, k1)
    x0, y0, x1, y1 = card_box("card_s22", text)
    m = 8.0
    band = ((L.wx + RECT_W / 2 > x0 - m) & (L.wx - RECT_W / 2 < x1 + m) &
            (L.wy + RECT_H / 2 > y0 - m) & (L.wy - RECT_H / 2 < y1 + m))
    band[BRIDGE_I] = False
    if band.any():
        k0 = max(k0, c0 + CARD_MIN_LEAD)
        avail = k0 - c0
        fb = min(0.45 * D, 0.6 * avail)
        start[band] = c0 + order[band] * (avail - fb)
        dur[band] = fb
    return start, dur, (k0, k1)


def s22(sh, t, sc):
    E = sh["events"]
    c0, c1 = E["collapse"]
    a = Ag.of(BRIDGE[MERGE_ROUND])
    caps = sh["captions"]
    start, dur, _ = s22_schedule(sh, caps[0]["text"] if caps else None)
    a.level = 1 - smooth_a((t - start) / dur)
    e = ease(prog(t, E["collapse"]))
    a.wx[BRIDGE_I] += (INIT.wx[50] - a.wx[BRIDGE_I]) * e
    a.wy[BRIDGE_I] += (INIT.wy[50] - a.wy[BRIDGE_I]) * e
    _bridge_look(a, 1 - e)
    a.rgb[BRIDGE_I] = WHITE_GOLD + (GOLD - WHITE_GOLD) * e
    a.level[BRIDGE_I] = 1.0
    a.glow[BRIDGE_I] *= 1 + 0.2 * math.sin(2 * math.pi * (t - c1) / 3.0) * smooth((t - c1) / 1.0)
    f = 1 - smooth(prog(t, E["black"]))
    a.level = a.level * f
    a.emit(sc, CAM1)
    sc.fade = f
    if "bg_out" in E:            # the world recedes first; the question fades last
        sc.bg_level = 1 - smooth(prog(t, E["bg_out"]))
    sc.axis_level = 1 - smooth(prog(t, (c0, c0 + 0.8)))


SHOTS = {f"S{k:02d}": fn for k, fn in enumerate(
    [None, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11,
     s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22]) if fn}


# ----------------------------------------------------------------------------- motion blur
MB_SHUTTER = 1.0     # fraction of the frame interval (360 deg: kills the strobing of the comb)
MB_MIN = 1.5         # design px per frame below which a frame is rendered without blur
MB_STEP = 6.0        # max raster px a rect moves within one sub-interval (<= the 6 px gap
                     # between rects, so neighbouring smears never overlap inside a sub-frame)
MB_LINE_STEP = 40.0  # lines are smeared analytically too and never overlap: coarser sampling
MB_MAX_SUB = 64      # only the S15 push-through needs this many (a few frames); every other
                     # move stays far below it


def motion_px(sa, sb):
    """Largest screen displacement (design px) of a visible rect, and of a line across itself,
    between two scenes of the same shot (primitives are matched by order; unmatched batches
    are ignored). Returns (rects, lines)."""
    D = 0.0
    for A, B in zip(sa.rects, sb.rects):
        if len(A["x"]) != len(B["x"]):
            continue
        vis = ((A["level"] > 0.02) | (B["level"] > 0.02)) & \
              (np.minimum(A["x"], B["x"]) < 1940) & (np.maximum(A["x"], B["x"]) > -20)
        if vis.any():
            D = max(D, float(np.hypot(B["x"] - A["x"], B["y"] - A["y"])[vis].max()))
    Dr, D = D, 0.0
    if len(sa.lines) == len(sb.lines):
        for la, lb in zip(sa.lines, sb.lines):
            if max(la[6], lb[6]) <= 0.02:
                continue
            # only motion across the line smears it (rays/threads growing along themselves don't)
            dx, dy = la[2] - la[0], la[3] - la[1]
            ln = math.hypot(dx, dy)
            nx, ny = (-dy / ln, dx / ln) if ln > 1e-6 else (1.0, 0.0)
            for i in (0, 2):
                mx, my = lb[i] - la[i], lb[i + 1] - la[i + 1]
                D = max(D, abs(mx * nx + my * ny) if ln > 1e-6 else math.hypot(mx, my))
    return Dr, D


# ----------------------------------------------------------------------------- chapters / HUD
CHAPTERS = {1: ("01 · 村子", "THE VILLAGE"), 2: ("02 · 规则", "THE RULE"),
            3: ("03 · 实验一：宽信任", "EXPERIMENT 1 · WIDE ε"),
            4: ("04 · 实验二：窄信任", "EXPERIMENT 2 · NARROW ε"),
            5: ("05 · 现实", "THE REAL WORLD"), 6: ("06 · 桥", "THE BRIDGE")}
CHAPTER_DELAY = {1: 0.45, 2: 0.35, 3: 0.2, 4: 0.25, 5: 0.75, 6: 0.85}   # after the transition
CHAPTER_LEN = 2.1
CAP_PRE, CAP_POST, CAP_SEP = 0.2, 0.35, 0.05   # caption shown this much before / after its slot


def _stats(x, skip=None):
    """(groups, spread) of a state exactly as hk_sim.report counts them. groups is None while
    hk_sim.groups sees a single chain that is wider than its own gap (a continuum, not a
    cluster): the HUD then shows 'GROUPS —' rather than a misleading 1."""
    o = np.delete(x, skip) if skip is not None else np.asarray(x)
    g, spread = len(hk_sim.groups(o)), float(o.max() - o.min())
    if g == 1 and spread >= 0.02:
        g = None
    return g, spread


def _groups_txt(g):
    return "—" if g is None else str(g)


EXP1 = [_stats(L.op) for L in WIDE]
EXP2 = [_stats(L.op) for L in NARROW]
BRS = [_stats(L.op, BRIDGE_I) for L in BRIDGE]
SPREAD1 = [s for _, s in EXP1]
SPREAD2 = [s for _, s in EXP2]
SPREADB = [s for _, s in BRS]
SPARK_N_EXP = max(len(WIDE), len(NARROW)) - 1        # both experiments share one x scale


def _round_state(t, ev):
    """(round entered, last completed round, progress toward the next, entry time, slot)."""
    k0, k1, p = round_pos(t, ev)
    wins = M.round_windows(ev)
    te, slot = None, 1.0
    for a, b, k, sl in wins:
        if k == k1 and t >= a:
            te, slot = a, sl
    return k1, k0, (p if k1 > k0 else 0.0), te, slot


def _flash(t, te, slot=1.0):
    if te is None or t < te:
        return 0.0
    return M.clamp01((slot - 0.25) / 0.5) * math.exp(-(t - te) / 0.22)


# ----------------------------------------------------------------------------- show
class Show:
    def __init__(self, tl):
        self.tl = tl
        self.shots = tl["shots"]
        self.duration = float(tl["duration"])
        self.fps = tl.get("fps", 30)
        self.by_id = {sh["id"]: sh for sh in self.shots}
        for sh in self.shots:
            if sh["id"] not in SHOTS:
                raise ValueError(f"no choreography for shot {sh['id']}")
        # S22's title card is the question itself; its caption would duplicate it on screen.
        self.cards = {"S04": CARD_S04}
        s22 = self.by_id["S22"]
        if s22["captions"]:
            self.cards["S22"] = s22["captions"][0]["text"]
        caps = []
        for sh in self.shots:
            for c in sh["captions"]:
                if c["show"] and c["text"] != self.cards.get(sh["id"]):
                    caps.append((c["start"], c["end"], c["text"], sh["id"]))
        caps.sort()
        self.captions = []
        for k, (s, e, txt, sid) in enumerate(caps):
            vs, ve = s - CAP_PRE, e + CAP_POST
            if k + 1 < len(caps):
                ve = min(ve, caps[k + 1][0] - CAP_PRE - CAP_SEP)
            self.captions.append((vs, ve, txt, sid))
        self.annotations = [(a, sh) for sh in self.shots for a in sh["annotations"]]
        self.chapters = []
        seen = set()
        for sh in self.shots:
            ch = sh["chapter"]
            if ch not in seen:
                seen.add(ch)
                a = sh["start"] + CHAPTER_DELAY.get(ch, 0.3)
                self.chapters.append((a, a + CHAPTER_LEN, CHAPTERS[ch]))
        B = self.by_id
        REF["S05_listen"] = B["S05"]["events"]["bracket_in"][1] + 0.4
        REF["S15_listen"] = B["S15"]["events"]["bracket_in"][1] + 0.15
        REF["S19_pulses"] = B["S19"]["events"]["threads"][1] - 0.1
        self.hud_windows = [("exp1", B["S08"]["start"] + 0.5, B["S10"]["end"]),
                            ("exp2", B["S11"]["start"] + 0.35, B["S14"]["start"] + 0.6),
                            ("bridge", B["S18"]["start"] + 1.0, B["S21"]["start"] + 0.45)]

    def shot_at(self, t):
        for sh in self.shots:
            if t < sh["end"]:
                return sh
        return self.shots[-1]

    def annotation_text(self, a, sh, t):
        """{r} is a live round counter, right-aligned in a fixed-width field (padded with
        spaces) so the centred annotation never shifts when the counter gains a digit."""
        txt = a["text"]
        c = a.get("counter")
        if "{r}" not in txt or not c:
            return txt
        if c.get("rounds") == "events":
            ks = sorted((v[0], int(k.split(":")[1])) for k, v in sh["events"].items() if k.startswith("round:"))
            r = ks[0][1] - 1
            for t0, k in ks:
                if t >= t0:
                    r = k
            widest = max([k for _, k in ks] + [ks[0][1] - 1])
        else:
            r = self.counter_value(c, t)
            widest = max(c["from"], c["to"])
        return txt.replace("{r}", str(r).rjust(len(str(widest))))

    @staticmethod
    def counter_value(c, t):
        return c["from"] + int((c["to"] - c["from"]) * prog(t, (c["t0"], c["t1"])))

    # -- HUD --------------------------------------------------------------------------
    def hud(self, t):
        """Top-right readout (text, params) from hk_sim states, or None. The round shown is
        the one being entered, like the bottom annotations' counters."""
        for name, a, b in self.hud_windows:
            if a <= t <= b:
                break
        else:
            return None
        sh = self.shot_at(t)
        sid, E = sh["id"], sh["events"]
        prm = dict(tin=t - a, tout=b - t, flash=0.0)
        if name == "exp1":
            if any(k.startswith("round:") for k in E):
                r, k, p, te, slot = _round_state(t, E)
                prm["flash"] = _flash(t, te, slot)
            else:
                r, k, p = 0, 0, 0.0
            g, sp = EXP1[r]
            prm["spark"] = dict(vals=SPREAD1, n_max=SPARK_N_EXP, k=k, p=p, scale=1.0)
            return f"ROUND {r:03d} · ε 0.30 · GROUPS {_groups_txt(g)} · SPREAD {sp:.2f}", prm
        if name == "exp2":
            ghost = (SPREAD1, SPARK_N_EXP)
            if sid == "S11":
                if t < E["narrow"][1]:
                    r = REWIND.round_at(ease_sine(prog(t, E["rewind"])))
                    g, sp = EXP1[r]
                    eps, vals, k = "0.30", SPREAD1, r
                else:
                    r, (g, sp), eps, vals, k = 0, EXP2[0], "0.20", SPREAD2, 0
                    prm["flash"] = _flash(t, E["narrow"][1])
                p = 0.0
            elif sid == "S12":
                r, k, p, te, slot = _round_state(t, E)
                g, sp = EXP2[r]
                eps, vals = "0.20", SPREAD2
                prm["flash"] = _flash(t, te, slot)
            else:                                   # S13 counter, S14 hold
                c = next((an.get("counter") for an in self.by_id["S13"]["annotations"] if an.get("counter")), None)
                r = self.counter_value(c, t) if c else len(NARROW) - 1
                g, sp = EXP2[-1]
                eps, vals, k, p = "0.20", SPREAD2, len(NARROW) - 1, 0.0
            prm["spark"] = dict(vals=vals, n_max=SPARK_N_EXP, k=k, p=p, scale=1.0, ghost=ghost)
            return f"ROUND {r:03d} · ε {eps} · GROUPS {_groups_txt(g)} · SPREAD {sp:.2f}", prm
        # bridge
        if sid == "S18":
            r, k, p = 0, 0, 0.0
            be = "0.40" if t >= E["widen"][1] else "0.20"
            prm["flash"] = _flash(t, E["widen"][1])
        elif sid in ("S19", "S20"):
            r, k, p, te, slot = _round_state(t, E)
            be = "0.40"
            prm["flash"] = _flash(t, te, slot)
        else:
            r, k, p, be = MERGE_ROUND, MERGE_ROUND, 0.0, "0.40"
        g, sp = BRS[r]
        if sid == "S20" and r == MERGE_ROUND:
            # the merge round: counts switch when the blocks actually touch
            a0, tm = E[f"round:{MERGE_ROUND}"][0], merge_time(sh)
            if t < tm:
                g, sp = BRS[r - 1]
            p = clamp01((t - a0) / max(tm - a0, 1e-6))
            if t >= tm:
                prm["flash"] = _flash(t, tm)
        prm["spark"] = dict(vals=SPREADB, n_max=len(BRIDGE) - 1, k=k, p=p, scale=0.4)
        return f"ROUND {r:03d} · ε 0.20 · BRIDGE ε {be} · GROUPS {_groups_txt(g)} · SPREAD {sp:.2f}", prm

    # -- one scene ------------------------------------------------------------------------
    def _shot_scene(self, sh, t):
        sc = Scene()
        SHOTS[sh["id"]](sh, t, sc)
        return sc

    def cam_at(self, t):
        """(camera, drift) at t: what the background's parallax layers follow."""
        sh = self.shot_at(t)
        return self._shot_scene(sh, t).cam, M.drift(t)

    def scene(self, t, full=True):
        """The scene at t. full=False: data plane only (motion-blur sub-frames)."""
        sh = self.shot_at(t)
        sc = self._shot_scene(sh, t)
        # afterimages: the same shot a moment earlier (skipped when nothing moves)
        pb = self._shot_scene(sh, t - M.TRAIL_DT[1])
        moving = False
        for b, p in zip(sc.rects, pb.rects):
            if b.get("life") and b.get("cam") and p.get("cam") and len(b["x"]) == len(p["x"]):
                wx0, wy0 = M._world(b, b["x"], b["y"])
                wx1, wy1 = M._world(p, p["x"], p["y"])
                if (np.hypot(wx1 - wx0, wy1 - wy0)[b["level"] > 0.02] > M.TRAIL_MIN_PX).any():
                    moving = True
                    break
        M.trails(sc, [self._shot_scene(sh, t - M.TRAIL_DT[0]), pb] if moving else [])
        amt = 0.4 if (sh["id"] == "S17" and t > sh["events"]["freeze"][0]) else 1.0
        M.life(sc, t, amt)
        pulse = M.round_pulse(sc, sh, t)
        d = M.drift(t)
        M.apply_drift(sc, d)
        if not full:
            return sc
        self._axis(sc, t, d, pulse)
        sc.spill = self._spill(sc)
        self._texts(sc, sh, t)
        return sc

    @staticmethod
    def _axis_y(sc, cam, d):
        """Screen y of the axis under camera cam: AXIS_DY below the row (following AXIS_Y_PARALLAX
        of the zoom), but never closer than AXIS_CLEAR (x zoom) below the lowest visible rect
        drawn with that same camera, so tall blocks under a push never stand on it."""
        z, cx = cam
        dz, dx, dy = d
        ys = Y0 + AXIS_DY * (1 + AXIS_Y_PARALLAX * (z - 1)) * dz + dy
        bot = None
        for b in sc.rects:
            c = b.get("cam")
            if not b.get("life") or c is None or abs(c[0] - z) > 1e-6 or abs(c[1] - cx) > 1e-6:
                continue
            m = b["level"] > 0.05
            if m.any():
                v = float((b["y"][m] + b["h"][m] / 2).max())
                bot = v if bot is None else max(bot, v)
        if bot is not None:
            ys = max(ys, bot + AXIS_CLEAR * z * dz)
        return ys

    def _axis(self, sc, t, d, pulse):
        """The opinion axis: one part per camera (S18's wipe shows two rect sets side by side,
        each with the axis under its own camera, split at the beam)."""
        ac = getattr(sc, "axis_cam", None)
        cam = (ac.z, ac.cx) if ac is not None else sc.cam
        split = getattr(sc, "axis_split", None)
        if split is not None:
            xb, cl, cr = split
            parts = [((cl.z, cl.cx), -1e9, xb), ((cr.z, cr.cx), xb, 1e9)]
        else:
            parts = [(cam, -1e9, 1e9)]
        parts = [(c, a, b, self._axis_y(sc, c, d)) for c, a, b in parts]
        draw = getattr(sc, "axis_draw", 1.0)
        level = getattr(sc, "axis_level", 1.0) * sc.fade
        xg = -300.0 + 2520.0 * ((t / 9.0) % 1.0)
        sc.axis = dict(parts=parts, drift=d, draw=draw, alpha=level, pulse=pulse, glint=xg)
        if level * draw <= 1e-3:
            return
        dz, dx, dy = d
        for (z, cx), c0, c1, ys in parts:
            xs = lambda v: ((float(X(v)) - cx) * z) * dz + 960.0 + dx
            half = (xs(1.0) - xs(0.0)) / 2 * draw
            for v, lab in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
                x = xs(v)
                if abs(x - xs(0.5)) <= half + 1.0 and -40 < x < 1960 and c0 <= x < c1:
                    sc.text("mono_at", lab, 0.30 * level, x=x, y=ys + 31.0, px=12, glow=0.6)

    @staticmethod
    def _spill(sc):
        xs, ys, cs = [], [], []
        for b in sc.rects:
            if not b.get("life"):
                continue
            m = b["level"] > 0.02
            xs.append(b["x"][m])
            ys.append(b["y"][m])
            cs.append(b["rgb"][m] * np.clip(b["level"][m], 0.0, 1.3)[:, None])
        if not xs:
            return None
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)

    def _texts(self, sc, sh, t):
        f = sc.fade
        for vs, ve, txt, sid in self.captions:
            if vs <= t <= ve:
                sc.text("caption", txt, f, tin=t - vs, tout=ve - t)
        for a, ash in self.annotations:
            if a["start"] <= t <= a["end"]:
                sc.text("annotation", self.annotation_text(a, ash, t), f,
                        tin=t - a["start"], tout=a["end"] - t)
        if sh["id"] == "S04":
            k0, k1 = sh["events"]["card"]
            if k0 <= t <= k1:
                sc.text("card_s04", self.cards["S04"], 1.0, tin=t - k0, tout=k1 - t)
        if sh["id"] == "S22" and "S22" in self.cards:
            k0, k1 = s22_schedule(sh, self.cards["S22"])[2]
            if k0 <= t <= k1:
                sc.text("card_s22", self.cards["S22"], f, tin=t - k0, tout=k1 - t)
        for a, b, labels in self.chapters:
            if a <= t <= b:
                sc.text("chapter", labels, f, tin=t - a, tout=b - t)
        h = self.hud(t)
        if h:
            txt, prm = h
            sc.text("hud", txt, f, **prm)
        dz, dx, dy = sc.drift
        for txt, x, y, al, px in sc.labels:
            sc.text("mono_at", txt, al * f, x=(x - 960.0) * dz + 960.0 + dx, y=(y - Y0) * dz + Y0 + dy,
                    px=px, glow=1.0)

    # -- motion blur ----------------------------------------------------------------------
    def motion_bounds(self, t, fps, scale=1.0):
        """Sub-interval boundaries of frame t's shutter (one frame, 360 deg, centred on t, never
        crossing a cut), or None when no visible rect or line moves more than MB_MIN design px.
        Enough sub-intervals that each one moves <= MB_STEP raster px, so the per-rect linear
        smear inside each sub-interval follows curved / eased paths closely."""
        sh = self.shot_at(t)
        half = 0.5 * MB_SHUTTER / fps
        ta = max(sh["start"], t - half)
        tb = min(sh["end"] - 1e-6, t + half)
        if tb <= ta:
            return None
        Dr, Dl = motion_px(self.scene(ta, False), self.scene(tb, False))
        if max(Dr, Dl) < MB_MIN:
            return None
        n = int(min(MB_MAX_SUB, max(1, math.ceil(Dr * scale / MB_STEP), math.ceil(Dl * scale / MB_LINE_STEP))))
        return [ta + k / n * (tb - ta) for k in range(n + 1)]
