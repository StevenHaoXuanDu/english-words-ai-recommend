"""Per-shot choreography for render.py.

Show(timeline).scene(t) -> render_core.Scene for absolute time t (seconds).
All times come from timeline.json (shot bounds + named events); all opinions and
positions come from hk_sim.run(). Nothing here hardcodes an absolute time.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hk_sim  # noqa: E402

from render_core import (GOLD, RECT_H, RECT_W, WHITE_GOLD, Y0, ROW_L, ROW_R,  # noqa: E402
                         Layout, Scene, X, card_box, layout_seq, opinion_rgb)

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
        self.rgb = None

    @staticmethod
    def of(L):
        return Ag(L.op, L.wx, L.wy)

    @staticmethod
    def mix(A, B, e):
        return Ag(A.op + (B.op - A.op) * e, A.wx + (B.wx - A.wx) * e, A.wy + (B.wy - A.wy) * e)

    @staticmethod
    def tween(A, B, p):
        """Round tween at linear progress p: opinion/x/colour ease over the first 70 % of the
        window, y (stacking) over the last 45 %, so converging rects meet side by side and only
        then stack (a simultaneous x/y lerp draws diagonal '/' bars mid-tween)."""
        ex, ey = ease(p / 0.70), ease((p - 0.55) / 0.45)
        return Ag(A.op + (B.op - A.op) * ex, A.wx + (B.wx - A.wx) * ex, A.wy + (B.wy - A.wy) * ey)

    def colours(self):
        if self.rgb is None:
            self.rgb = opinion_rgb(self.op)
        return self.rgb

    def emit(self, sc, cam):
        sc.add_rects(cam.x(self.wx), cam.y(self.wy), RECT_W * cam.z, RECT_H * cam.z,
                     self.colours(), self.level, self.glow)


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


# ----------------------------------------------------------------------------- shots
def s01(sh, t, sc):
    E = sh["events"]
    a = Ag.of(INIT)
    a.level[:] = 0.0
    a.level[50] = smooth(prog(t, E["rect_in"]))
    a.glow[50] = 1.0 + 0.22 * math.sin(2 * math.pi * (t - sh["start"]) / 3.0)
    a.emit(sc, CAM1)


def s02(sh, t, sc):
    t0, t1 = sh["events"]["light_up"]
    fade = 0.4
    d = np.abs(np.arange(N) - 49.5)
    start = t0 + (d - 0.5) / 49.0 * max(t1 - t0 - fade, 1e-6)
    a = Ag.of(INIT)
    a.level = smooth_a((t - start) / fade)
    a.level[50] = 1.0
    a.emit(sc, CAM1)


S03_ZOOM, S03_RAMP, S03_FROM, S03_TO = 0.6, 0.4, 0.36, 0.64


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


def s04(sh, t, sc):
    E = sh["events"]
    a = Ag.of(INIT)
    a.level[:] = 1 - 0.7 * smooth(prog(t, E["dim"])) + 0.7 * smooth(prog(t, E["undim"]))
    a.emit(sc, CAM1)
    sc.text("card_s04", CARD_S04, window_fade(t, E["card"], 0.5))


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
    e = ease_sine(prog(t, E["zoom_in"]))      # sine: half the peak speed of easeInOutCubic
    cam = Cam(1 + (S05_CAM.z - 1) * e, 960 + (S05_CAM.cx - 960) * e)
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
    a.emit(sc, cam)
    bracket(sc, cam, X0[DEMO_I], eps, o, INIT, snap=settle(pw, head=False))
    demo_marker(sc, cam, a.wx[DEMO_I], a.wy[DEMO_I], f)
    sc.zoom = cam.z


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


def s06(sh, t, sc):
    E = sh["events"]
    cam = S05_CAM
    m0, m1 = E["move"]
    a = demo_move(prog(t, E["move"]))
    ins = inside(X0, X0[DEMO_I], 0.20)
    focus_look(a, 1.0, 1 - 0.8 * (1 - ins))
    a.emit(sc, cam)
    demo_marker(sc, cam, a.wx[DEMO_I], a.wy[DEMO_I], 1.0)
    bracket(sc, cam, X0[DEMO_I], 0.20, 1.0, INIT)
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


def s07(sh, t, sc):
    E = sh["events"]
    ez = ease_sine(prog(t, E["zoom_out"]))
    cam = Cam(S05_CAM.z + (1 - S05_CAM.z) * ez, S05_CAM.cx + (960 - S05_CAM.cx) * ez)
    pr = prog(t, E["reset_demo"])
    er = ease(pr)
    if t < E["round:1"][0]:
        a = demo_move(pr, back=True)
    elif t < E["rewind"][0]:
        a = Ag.tween(INIT, DEMO1, prog(t, E["round:1"]))
    else:
        a = Ag.mix(DEMO1, INIT, ease(prog(t, E["rewind"])))
    focus_look(a, 1 - er, 1 - 0.8 * (1 - er) * (1 - inside(X0, X0[DEMO_I], 0.20)))
    a.emit(sc, cam)
    if t < E["round:1"][0]:
        demo_marker(sc, cam, a.wx[DEMO_I], a.wy[DEMO_I], 1 - smooth(pr / 0.7))
    bracket(sc, cam, X0[DEMO_I], 0.20, 1 - er, INIT)
    pf = prog(t, E["flash"])
    if 0 < pf < 1:
        o = 0.12 * math.sin(math.pi * pf)
        for i in range(N):
            bracket(sc, cam, X0[i], 0.20, 1.0, INIT, alpha=o, glow=0.0,
                    half=46.0 + 34.0 * ((i * 0.6180339887) % 1.0))
    sc.zoom = cam.z


def s08(sh, t, sc):
    E = sh["events"]
    pin = prog(t, E["bracket_in"])
    ein, eout = ease(pin), ease(prog(t, E["bracket_out"]))
    a = Ag.of(INIT)
    a.level = 1 - 0.65 * ein * (1 - eout) * (1 - inside(a.op, X0[50], 0.30))
    a.emit(sc, CAM1)
    bracket(sc, CAM1, X0[50], 0.30 * ein, smooth(pin * 3) * (1 - eout), INIT,
            snap=settle(pin, head=False))


def _rounds(seq, sh, t):
    k0, k1, p = round_pos(t, sh["events"])
    return Ag.tween(seq[k0], seq[k1], p)


def s09(sh, t, sc):
    _rounds(WIDE, sh, t).emit(sc, CAM1)


def s10(sh, t, sc):
    _rounds(WIDE, sh, t).emit(sc, CAM1)
    sc.glow_gain = 1.0 + 0.4 * bump(prog(t, sh["events"]["glow"]))


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
        for k in range(len(seq) - 1, 0, -1):
            A, B = seq[k], seq[k - 1]
            dy = B.wy - A.wy
            if np.abs(dy).max() > 0.5:
                moves.append((0.0, 0.0, dy, np.abs(dy).max()))
            dx = B.wx - A.wx
            if np.abs(dx).max() > 0.5 or np.abs(B.op - A.op).max() > 1e-9:
                moves.append((B.op - A.op, dx, 0.0, np.abs(dx).max()))
        lens = [max(m[3], 20.0) ** 0.6 for m in moves]
        wins, t = [], 0.0
        for k, L in enumerate(lens):
            if k:
                t -= self.OVERLAP * min(L, lens[k - 1])
            wins.append((t, t + L))
            t += L
        self.moves = [(m[0], m[1], m[2], (a / t, b / t)) for m, (a, b) in zip(moves, wins)]

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


def s11(sh, t, sc):
    E = sh["events"]
    a = REWIND.at(ease_sine(prog(t, E["rewind"])))
    pin = prog(t, E["bracket_in"])
    ein = ease(pin)
    pn = prog(t, E["narrow"])
    eps = 0.30 - 0.10 * ease(pn)
    a.level = 1 - 0.65 * ein * (1 - inside(a.op, X0[50], eps))
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


def s13(sh, t, sc):
    Ag.of(NSEQ[-1]).emit(sc, CAM1)
    sc.glow_gain = 1.0 + 0.03 * math.sin(2 * math.pi * (t - sh["start"]) / 4.0)


def s14(sh, t, sc):
    cam = Cam(1.0 + 0.3 * ease(prog(t, sh["events"]["push"])), float(X(0.5)))
    Ag.of(NSEQ[-1]).emit(sc, cam)
    sc.zoom = cam.z


def s15(sh, t, sc):
    E = sh["events"]
    p = prog(t, E["reset"])
    old = Ag.of(NSEQ[-1])
    old.level[:] = 1 - smooth(p / 0.6)
    cam_old = Cam(1.3 - 0.15 * ease(p), float(X(0.5)))
    old.emit(sc, cam_old)
    new = Ag.of(INIT)
    new.level[:] = smooth((p - 0.35) / 0.65)
    new.emit(sc, CAM1)
    pin = prog(t, E["bracket_in"])
    bracket(sc, CAM1, X0[50], 0.30 * ease(pin), smooth(pin * 3), INIT, snap=settle(pin, head=False))
    sc.zoom = cam_old.z + (1 - cam_old.z) * smooth(p)


S16_EPS = [0.30, 0.25, 0.20, 0.15, 0.10]
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


def _s16_state(e):
    """(eps, level per agent, x offset per agent) after beat progress e[0..3]."""
    eps = S16_EPS[0] + sum((S16_EPS[k] - S16_EPS[k - 1]) * e[k - 1] for k in range(1, 5))
    dim = S16_DIM[0] + sum((S16_DIM[k] - S16_DIM[k - 1]) * e[k - 1] for k in range(1, 5))
    ins = inside(X0, X0[50], eps)
    level = ins + (1 - ins) * dim
    level[S16_BLOCKED] *= 1 - e[2]
    return eps, level, S16_DRIFT * e[3]


def s16(sh, t, sc):
    E = sh["events"]
    ps = [prog(t, E[f"beat:{k}"]) for k in range(1, 5)]
    eps, level, dx = _s16_state([ease(p) for p in ps])
    a = Ag.of(INIT)
    a.level, a.wx = level, a.wx + dx
    a.emit(sc, CAM1)
    bracket(sc, CAM1, X0[50], eps, 1.0, INIT, snap=math.prod(settle(p) for p in ps),
            pos=(INIT.op, a.wx))


def s17(sh, t, sc):
    E = sh["events"]
    er = ease(prog(t, E["restore"]))
    eps, level, dx = _s16_state([1.0] * 4)
    a = _rounds(NSEQ, sh, t)
    a.level = level + (1 - level) * er
    a.wx = a.wx + dx * (1 - er)
    tf = E["freeze"][0]
    if t >= tf:
        u = (t - tf) / 0.15
        f = bump(u)
        sc.glow_gain = 1.0 - 0.3 * smooth(u) + 0.9 * f
        a.level = a.level * (1 + 0.25 * f)
    a.emit(sc, CAM1)
    bracket(sc, CAM1, X0[50], eps, 1 - er, INIT, pos=(INIT.op, a.wx))


def _bridge_look(a, amount=1.0):
    rgb = a.colours().copy()
    rgb[BRIDGE_I] = rgb[BRIDGE_I] + (WHITE_GOLD - rgb[BRIDGE_I]) * amount
    a.rgb = rgb
    a.glow[BRIDGE_I] = 1.0 + 0.6 * amount


def s18(sh, t, sc):
    E = sh["events"]
    p = smooth(prog(t, E["reset"]))
    old = Ag.of(NSEQ[3])
    old.level[:] = 1 - p
    old.emit(sc, CAM1)
    new = Ag.of(BRIDGE[0])
    new.level[:] = p
    _bridge_look(new, ease(prog(t, E["highlight"])))
    new.emit(sc, CAM1)
    sc.glow_gain = 0.7 + 0.3 * p
    pin = prog(t, E["bracket_in"])
    eps = 0.20 * ease(pin) + 0.20 * ease(prog(t, E["widen"]))
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


def s20(sh, t, sc):
    E = sh["events"]
    k0, k1, e = round_pos(t, E)
    a = Ag.tween(BRIDGE[k0], BRIDGE[k1], e)
    _bridge_look(a)
    a.emit(sc, CAM1)
    threads(sc, CAM1, a, 1.0, 0.5 * (1 - smooth(prog(t, E[f"round:{MERGE_ROUND}"]))))


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
    sc.add_rects(xs, ys, RECT_W * k, RECT_H * k, rgb, lv, gl)
    gap = 3.0
    bx, cy = xs[BRIDGE_I], ys[BRIDGE_I]
    for a_x, b_x in ((xs[LEFT].max() + RECT_W * k / 2 + gap, bx - RECT_W * k / 2 - gap),
                     (xs[RIGHT].min() - RECT_W * k / 2 - gap, bx + RECT_W * k / 2 + gap)):
        if t < b0:
            sc.line(a_x, cy, b_x, cy, 1.0, GOLD, 0.6 * o, 0.5)
        else:                                      # snap: halves recoil to their ends and fade
            u = ease_out(prog(t, (b0, b1 + 0.15)))
            mid = (a_x + b_x) / 2
            al = 0.6 * o * (1 - smooth(prog(t, (b0, b1 + 0.25))))
            sc.line(a_x, cy, a_x + (mid - a_x) * (1 - u), cy, 1.0, GOLD, al, 0.5)
            sc.line(b_x, cy, b_x + (mid - b_x) * (1 - u), cy, 1.0, GOLD, al, 0.5)


def s21(sh, t, sc):
    E = sh["events"]
    a = Ag.of(BRIDGE[MERGE_ROUND])
    _bridge_look(a)
    a.level[:] = 1 - 0.6 * smooth(prog(t, E["dim"])) + 0.6 * smooth(prog(t, E["undim"]))
    a.emit(sc, CAM1)
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


SHOTS = {f"S{k:02d}": fn for k, fn in enumerate(
    [None, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11,
     s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22]) if fn}


# ----------------------------------------------------------------------------- motion blur
MB_SHUTTER = 1.0     # fraction of the frame interval (360 deg: kills the strobing of the comb)
MB_MIN = 1.5         # design px per frame below which a frame is rendered without blur
MB_STEP = 6.0        # max raster px a rect moves within one sub-interval (<= the 6 px gap
                     # between rects, so neighbouring smears never overlap inside a sub-frame)
MB_LINE_STEP = 40.0  # lines are smeared analytically too and never overlap: coarser sampling
MB_MAX_SUB = 24


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


# ----------------------------------------------------------------------------- show
class Show:
    def __init__(self, tl):
        self.tl = tl
        self.shots = tl["shots"]
        self.duration = float(tl["duration"])
        self.fps = tl.get("fps", 30)
        for sh in self.shots:
            if sh["id"] not in SHOTS:
                raise ValueError(f"no choreography for shot {sh['id']}")
        # S22's title card is the question itself; its caption would duplicate it on screen.
        self.cards = {"S04": CARD_S04}
        s22 = next(s for s in self.shots if s["id"] == "S22")
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
            vs, ve = s - 0.1, e + 0.3
            if k + 1 < len(caps):
                ve = min(ve, caps[k + 1][0] - 0.1)
            self.captions.append((vs, ve, txt, sid))
        self.annotations = [(a, sh) for sh in self.shots for a in sh["annotations"]]

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
            r = c["from"] + int((c["to"] - c["from"]) * prog(t, (c["t0"], c["t1"])))
            widest = max(c["from"], c["to"])
        return txt.replace("{r}", str(r).rjust(len(str(widest))))

    def scene(self, t):
        sh = self.shot_at(t)
        sc = Scene()
        sc.fade = 1.0
        SHOTS[sh["id"]](sh, t, sc)
        for vs, ve, txt, sid in self.captions:
            if vs <= t <= ve:
                sc.text("caption", txt, window_fade(t, (vs, ve), 0.3) * sc.fade)
        for a, ash in self.annotations:
            if a["start"] <= t <= a["end"]:
                sc.text("annotation", self.annotation_text(a, ash, t),
                        window_fade(t, (a["start"], a["end"]), 0.4) * sc.fade)
        if sh["id"] == "S22" and "S22" in self.cards:
            win = s22_schedule(sh, self.cards["S22"])[2]
            sc.text("card_s22", self.cards["S22"], window_fade(t, win, 0.5) * sc.fade)
        return sc

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
        Dr, Dl = motion_px(self.scene(ta), self.scene(tb))
        if max(Dr, Dl) < MB_MIN:
            return None
        n = int(min(MB_MAX_SUB, max(1, math.ceil(Dr * scale / MB_STEP), math.ceil(Dl * scale / MB_LINE_STEP))))
        return [ta + k / n * (tb - ta) for k in range(n + 1)]
