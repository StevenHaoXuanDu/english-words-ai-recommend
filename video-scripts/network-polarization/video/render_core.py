"""Low-level drawing for render.py: colours, stacking layout, rasteriser, glow, text.

Everything here works in 1920x1080 "design" pixels; a Raster built with scale s
multiplies every coordinate by s, so previews render natively at lower resolution.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

W, H = 1920, 1080
Y0 = 500.0                      # row centreline
RECT_W, RECT_H = 10.0, 28.0
COL_PITCH, ROW_PITCH = 14.0, 32.0
GROUP_TOL = 0.0015              # opinions closer than this stack into one block
ROW_L = 160.0 - RECT_W / 2 - 8  # bracket clip: 8 px beyond the first / last rect
ROW_R = 1760.0 + RECT_W / 2 + 8

GLOW_SIGMA_IN, GLOW_GAIN_IN = 6.0, 0.55     # "blur 12 px, 35 %" from the script, lifted
GLOW_SIGMA_OUT, GLOW_GAIN_OUT = 26.0, 0.24  # wide faint bloom
BODY_KNEE = 0.93                            # rect body: brightest channel rolls off above this
GLOW_KNEE, GLOW_CEIL = 0.45, 0.80           # soft knee on the summed glow (brightest channel):
                                            # one rect, a row or a block stays below the knee
                                            # (glow <= 0.49); dense mid-tween piles roll off
                                            # toward 0.80, hue kept, so they stay gold, not white


def X(x):
    """Opinion -> design-pixel x."""
    return 160.0 + 1600.0 * np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------- colour
def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def srgb_to_oklab(c):
    return np.cbrt(_to_lin(c) @ _M1.T) @ _M2.T


def oklab_to_srgb(lab):
    lms = (lab @ np.linalg.inv(_M2).T) ** 3
    return _to_srgb(lms @ np.linalg.inv(_M1).T)


TEAL, GOLD, VERM = hex_rgb("#5BB5A8"), hex_rgb("#E8C066"), hex_rgb("#D4573B")
WHITE_GOLD = hex_rgb("#FFF4D6")
CAPTION_RGB = hex_rgb("#F3E9D2")

_LUT_N = 2049
_t = np.linspace(0.0, 1.0, _LUT_N)
_lt, _lg, _lv = srgb_to_oklab(TEAL), srgb_to_oklab(GOLD), srgb_to_oklab(VERM)
_lab = np.where((_t <= 0.5)[:, None],
                _lt + (_lg - _lt) * np.clip(_t / 0.5, 0, 1)[:, None],
                _lg + (_lv - _lg) * np.clip((_t - 0.5) / 0.5, 0, 1)[:, None])
LUT = oklab_to_srgb(_lab)
LUT[0], LUT[(_LUT_N - 1) // 2], LUT[-1] = TEAL, GOLD, VERM


def opinion_rgb(x):
    i = np.clip(np.rint(np.asarray(x) * (_LUT_N - 1)).astype(int), 0, _LUT_N - 1)
    return LUT[i]


# 8-bit sRGB <-> linear light, for averaging motion-blur sub-frames
SRGB_TO_LIN = _to_lin(np.arange(256) / 255.0).astype(np.float32)
_LIN_N = 16384
LIN_TO_SRGB = np.clip(np.rint(_to_srgb(np.linspace(0.0, 1.0, _LIN_N)) * 255.0), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- layout
ATTACH_DX = (6.0, 12.0)   # straggler parked at a block's free slot: outward offset range (px)
ATTACH_LIFT = 5.0         # ... and lift above that slot (px), so it reads as not yet inside
TIE = 1e-4                # px^2 per (rank - slot)^2: breaks exact ties in slot assignment
SWITCH = (6 * ROW_PITCH) ** 2   # px^2 per change of target column along one source column
COHERE_MAX = 30000              # skip the re-deal when a source column has more labelings than this


def _labelings(counts):
    """Every sequence over len(counts) labels that uses label c exactly counts[c] times."""
    n = sum(counts)
    seq = [0] * n
    left = list(counts)

    def rec(i):
        if i == n:
            yield tuple(seq)
            return
        for c in range(len(left)):
            if left[c]:
                left[c] -= 1
                seq[i] = c
                yield from rec(i + 1)
                left[c] += 1
    return rec(0)


def _cohere(pick, S, px, py):
    """Keep each source column together. A source column is the members of a group drawn at
    the same x just before (a single-column stack, or one column of a block). When the
    assignment splits one over several block columns, the split is dealt out row by row in
    alternation (a 10-stack straddling two columns: left, right, left, ...), which during the
    x-first half of a round tween shows as a zig-zag 'zipper'. Here the rects of the source
    column are re-dealt among those same target columns, as many to each as before (so the
    horizontal movement is unchanged), choosing the deal with the fewest changes of target
    column walking the source column top to bottom, SWITCH px^2 each, on top of the squared
    vertical moves. Inside every target column touched, rows go to its rects in the order of
    their previous height, so nothing passes through anything while stacking."""
    pick = np.array(pick)
    k = len(pick)
    tx = np.round(S[:, 0], 3)
    done = np.zeros(k, bool)
    for i in range(k):
        if done[i]:
            continue
        src = np.flatnonzero(np.abs(px - px[i]) < 0.5)
        done[src] = True
        if len(src) < 2:
            continue
        cols = np.unique(tx[pick[src]])
        if len(cols) < 2:
            continue
        mem = src[np.argsort(py[src], kind="stable")]                    # top to bottom
        counts = [int((tx[pick[mem]] == c).sum()) for c in cols]
        if math.factorial(len(mem)) // math.prod(math.factorial(c) for c in counts) > COHERE_MAX:
            continue
        cur_ty = S[pick, 1]
        insrc = np.zeros(k, bool)
        insrc[mem] = True
        others = [np.flatnonzero((tx[pick] == c) & ~insrc) for c in cols]
        slots = [np.flatnonzero(tx == c)[np.argsort(S[tx == c, 1], kind="stable")] for c in cols]
        best = None
        for lab in _labelings(counts):
            lab = np.array(lab)
            cost = SWITCH * int((lab[1:] != lab[:-1]).sum())
            deal = []
            for ci in range(len(cols)):
                m = np.concatenate([others[ci], mem[lab == ci]])
                m = m[np.lexsort((cur_ty[m], py[m]))]
                cost += float(((S[slots[ci], 1] - py[m]) ** 2).sum())
                deal.append(m)
            if best is None or cost < best[0] - 1e-6:
                best = (cost, deal)
        for ci, m in enumerate(best[1]):
            pick[m] = slots[ci]
    return pick


def _slot_xy(j, k, xm, inward):
    """Design-px centre of grid slot j of a k-member block centred on xm. Slots fill column-major,
    bottom row first, starting from the block's outer column (inward=+1: left column first, the
    partial column ends up on the right; -1 mirrored), so a block's free slots face the middle
    of the row, where newcomers arrive from. j >= k gives the block's next free slots."""
    rows, cols = min(10, k), math.ceil(k / 10)
    c, r = divmod(j, 10)
    ci = c if inward > 0 else cols - 1 - c
    return xm + (ci - (cols - 1) / 2) * COL_PITCH, Y0 + ((rows - 1) / 2 - r) * ROW_PITCH


class Layout:
    """Screen layout of one opinion vector: groups stack into centred column-major grids.

    prev (a Layout) or prev_pos ((wx, wy) arrays) is where the rects are drawn just before this
    state. Each group's grid slots then go to its members by minimum total squared movement
    (Hungarian assignment), so members already in a block keep their slots from round to round
    and a newcomer takes the free slot nearest to it; _cohere then keeps each source column
    (a stack, or a block column) in as few straight runs as possible when it lands in several
    block columns. Build a round sequence with layout_seq().

    Two drawing rules keep overlaps honest and legible (opinions themselves are untouched):
    * a single rect whose opinion falls inside a multi-column block's footprint is parked at the
      block's next free slot, offset outward toward its true side and lifted a few px: visibly
      separate from the block until it really joins;
    * a run of overlapping single-column stacks with mixed odd/even heights is drawn on one shared
      row grid (the parity with fewer votes shifts up half a row; single rects touching the run
      vote odd, as they sit on the row line), instead of a brick-bond of staggered rows with
      pinholes. Horizontal positions are not changed, so no spread is misrepresented. Runs that
      lean on a multi-column block are left alone (aligning them could hide rects exactly
      behind a block column).
    """

    def __init__(self, op, prev=None, prev_pos=None):
        op = np.asarray(op, dtype=np.float64)
        self.op = op
        n = len(op)
        if prev is not None:
            prev_pos = (prev.wx, prev.wy)
        order = np.argsort(op, kind="stable")
        groups, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if op[b] - op[a] <= GROUP_TOL:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        self.wx = np.empty(n)
        self.wy = np.empty(n)
        self.gsize = np.empty(n, dtype=int)   # size of the group each agent stacks in
        self.attached = np.zeros(n, bool)     # straggler drawn parked at a block's free slot

        geo = []                               # [members, k, centre x, inward, row shift]
        for g in groups:
            g = sorted(int(i) for i in g)
            m = float(op[g].mean())
            geo.append([g, len(g), float(X(m)), 1 if m <= 0.5 else -1, 0.0])
        multi = [G for G in geo if G[1] > 10]

        # one shared row grid inside each run of overlapping single-column stacks
        stacks = [G for G in geo if 2 <= G[1] <= 10]
        runs, run = [], []
        for G in stacks:
            if run and G[2] - run[-1][2] < RECT_W + 1.0:
                run.append(G)
            else:
                if len(run) > 1:
                    runs.append(run)
                run = [G]
        if len(run) > 1:
            runs.append(run)
        for run in runs:
            if any(abs(G[2] - B[2]) < (math.ceil(B[1] / 10) - 1) / 2 * COL_PITCH + RECT_W
                   for G in run for B in multi):
                continue                       # leans on a big block: leave as is
            odd = [G for G in run if G[1] % 2]
            even = [G for G in run if not G[1] % 2]
            if odd and even:
                # single rects touching the run sit on the row line: they vote for odd rows
                touch = sum(1 for G in geo if G[1] == 1 and
                            any(abs(G[2] - S[2]) < RECT_W + 1.0 for S in run))
                for G in (odd if len(odd) + touch < len(even) else even):
                    G[4] = -ROW_PITCH / 2

        self.groups = []          # (centre x, half width, half height, members)
        for g, k, xm, inward, dy in geo:
            self.gsize[g] = k
            S = np.array([_slot_xy(j, k, xm, inward) for j in range(k)])
            S[:, 1] += dy
            if k == 1:
                pick = [0]
            else:
                # canonical order: from the outer side inward (ties: by index)
                key = np.array(g, float) if prev_pos is None else prev_pos[0][g]
                rank = np.empty(k)
                rank[np.lexsort((np.array(g), inward * key))] = np.arange(k)
                if prev_pos is None:
                    pick = rank.astype(int)
                else:
                    px, py = prev_pos[0][g], prev_pos[1][g]
                    cost = ((px[:, None] - S[None, :, 0]) ** 2 + (py[:, None] - S[None, :, 1]) ** 2
                            + TIE * (rank[:, None] - np.arange(k)[None, :]) ** 2)
                    _, pick = linear_sum_assignment(cost)
                    pick = _cohere(pick, S, px, py)
            self.wx[g] = S[pick, 0]
            self.wy[g] = S[pick, 1]
            rows, cols = min(10, k), math.ceil(k / 10)
            self.groups.append((xm, (cols - 1) / 2 * COL_PITCH + RECT_W / 2,
                                (rows - 1) / 2 * ROW_PITCH + RECT_H / 2 + abs(dy), g))

        # stragglers inside a multi-column block's footprint: park at its next free slot
        used = {}
        for gi, (g, k, xm, inward, dy) in enumerate(geo):
            if k != 1:
                continue
            for bi, (_, bk, bx, binw, _) in enumerate(geo):
                hw = (math.ceil(bk / 10) - 1) / 2 * COL_PITCH + RECT_W / 2
                if bk > 10 and abs(xm - bx) < hw:
                    j = bk + used.get(bi, 0)
                    used[bi] = used.get(bi, 0) + 1
                    sx, sy = _slot_xy(j, bk, bx, binw)
                    side = math.copysign(1.0, xm - bx) if xm != bx else binw
                    i = g[0]
                    self.wx[i] = sx + side * min(max(abs(xm - bx), ATTACH_DX[0]), ATTACH_DX[1])
                    self.wy[i] = sy - ATTACH_LIFT
                    self.attached[i] = True
                    self.groups[gi] = (float(self.wx[i]), RECT_W / 2, RECT_H / 2, g)
                    break


def layout_seq(ops, first=None):
    """Layouts of consecutive states, each assigned against the one before (see Layout).
    `first` (a Layout of ops[0]) lets a sequence continue from a state already on screen."""
    ops = [np.asarray(x, dtype=np.float64) for x in ops]
    if first is not None:
        assert np.array_equal(first.op, ops[0]), "sequence must start from first's state"
        out = [first]
    else:
        out = [Layout(ops[0])]
    for x in ops[1:]:
        out.append(Layout(x, prev=out[-1]))
    return out


# ----------------------------------------------------------------------------- scene
class Scene:
    """Primitives in design pixels (after camera). Raster draws them; render_fx draws the
    background (cam, drift, axis, fx, spill) and render_text the texts."""

    def __init__(self):
        self.rects = []      # dict(x, y, w, h, rgb(n,3), level(n), glow(n), cam, life, smear)
        self.lines = []      # (x0, y0, x1, y1, width, rgb, alpha, glow)
        self.dots = []       # (x, y, radius, rgb, alpha, glow)
        self.zoom = 1.0      # scales glow radius
        self.glow_gain = 1.0
        self.texts = []      # (kind, text, opacity, params)
        self.cam = (1.0, 960.0)          # main camera (zoom, centre x) of the shot
        self.drift = (1.0, 0.0, 0.0)     # global camera drift (zoom, dx, dy), applied last
        self.fx = []         # background light effects (render_fx.FX)
        self.axis = None
        self.bg_level = 1.0
        self.fade = 1.0

    def add_rects(self, x, y, w, h, rgb, level, glow, cam=None, life=False, smear=None):
        """cam: (zoom, centre x) the batch was projected with (lets trails remove camera motion).
        life: the batch gets the ambient breathing / shimmer. smear: fixed (sx, sy) smear in
        design px (motion trails); such batches keep it under motion blur."""
        n = len(x)
        self.rects.append(dict(x=np.array(x, float), y=np.array(y, float),
                               w=np.array(np.broadcast_to(np.asarray(w, float), (n,))),
                               h=np.array(np.broadcast_to(np.asarray(h, float), (n,))),
                               rgb=np.asarray(rgb, float).reshape(n, 3),
                               level=np.array(np.broadcast_to(np.asarray(level, float), (n,))),
                               glow=np.array(np.broadcast_to(np.asarray(glow, float), (n,))),
                               cam=cam, life=life, fixed=smear is not None))
        if smear is not None:
            self.rects[-1]["sx"], self.rects[-1]["sy"] = smear

    def line(self, x0, y0, x1, y1, width, rgb, alpha, glow=0.0):
        if alpha > 1e-3:
            self.lines.append((x0, y0, x1, y1, width, np.asarray(rgb, float), alpha, glow))

    def dot(self, x, y, r, rgb, alpha, glow=1.0):
        if alpha > 1e-3:
            self.dots.append((x, y, r, np.asarray(rgb, float), alpha, glow))

    def text(self, kind, text, opacity, **params):
        if opacity > 1e-3:
            self.texts.append((kind, text, float(min(1.0, opacity)), params))


# ----------------------------------------------------------------------------- raster
def _S(x):
    """Integral of clamp(x, 0, 1)."""
    return np.where(x <= 0.0, 0.0, np.where(x >= 1.0, x - 0.5, 0.5 * x * x))


def _span(a, b, n, h=0.0):
    """Pixel range [i0, i1) and per-pixel coverage of the interval [a, b], clipped to [0, n).
    h > 0: time-averaged coverage while the interval slides uniformly over [-h, +h]
    (analytic box-blur along the motion = exact motion blur for linear motion)."""
    i0, i1 = max(int(math.floor(a - h)), 0), min(int(math.ceil(b + h)), n)
    if i1 <= i0:
        return None
    p = np.arange(i0, i1, dtype=np.float32)
    if h < 1e-3:
        c = np.minimum(p + 1.0, b) - np.maximum(p, a)
    else:
        c = (_S(b - p + h) - _S(b - p - h) - _S(a - p + h) + _S(a - p - h)) / (2.0 * h)
    return i0, i1, np.clip(c, 0.0, 1.0).astype(np.float32)


def _box_cov(x0, y0, x1, y1, w, h, hx=0.0, hy=0.0):
    sx, sy = _span(x0, x1, w, hx), _span(y0, y1, h, hy)
    if sx is None or sy is None:
        return None
    return sx[0], sx[1], sy[0], sy[1], sy[2][:, None] * sx[2][None, :]


SEG_SMEAR_STEP = 0.75   # px between the sampled positions of a smeared diagonal segment


def _seg_cov(x0, y0, x1, y1, width, w, h, hx=0.0, hy=0.0, gamma=1.0):
    """Anti-aliased coverage of a segment of the given width (distance field).
    (hx, hy) != 0: time-averaged coverage while the segment slides uniformly from -(hx, hy) to
    +(hx, hy) (signed): motion blur for diagonal lines, sampled every SEG_SMEAR_STEP px.
    gamma > 1 averages the samples as cov**gamma (then takes the root): coverage becomes an
    sRGB value, and motion-blur sub-frames are averaged in linear light, so this keeps the smear
    of a thin line even across sub-frame boundaries instead of dimming there."""
    hw = width / 2.0
    pad = hw + 1.5
    ax, ay = abs(hx), abs(hy)
    i0 = max(int(math.floor(min(x0, x1) - ax - pad)), 0)
    i1 = min(int(math.ceil(max(x0, x1) + ax + pad)), w)
    j0 = max(int(math.floor(min(y0, y1) - ay - pad)), 0)
    j1 = min(int(math.ceil(max(y0, y1) + ay + pad)), h)
    if i1 <= i0 or j1 <= j0:
        return None
    px = np.arange(i0, i1, dtype=np.float32)[None, :] + 0.5
    py = np.arange(j0, j1, dtype=np.float32)[:, None] + 0.5
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    m = int(min(48, max(1, math.ceil(2.0 * math.hypot(hx, hy) / SEG_SMEAR_STEP))))
    cov = np.zeros((j1 - j0, i1 - i0), np.float32)
    for k in range(m):
        o = (2.0 * (k + 0.5) / m - 1.0) if m > 1 else 0.0
        sx, sy = x0 + o * hx, y0 + o * hy
        if L2 < 1e-9:
            d = np.hypot(px - sx, py - sy)
        else:
            u = np.clip(((px - sx) * dx + (py - sy) * dy) / L2, 0.0, 1.0)
            d = np.hypot(px - (sx + u * dx), py - (sy + u * dy))
        c = np.clip(hw + 0.5 - d, 0.0, 1.0) if width >= 1.0 else width * np.clip(1.0 - d, 0.0, 1.0)
        cov += c if (m == 1 or gamma == 1.0) else c ** gamma
    if m > 1:
        cov *= 1.0 / m
        if gamma != 1.0:
            cov **= 1.0 / gamma
    return i0, i1, j0, j1, cov


def _next_fast(n):
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


class Raster:
    def __init__(self, scale=1.0):
        self.s = scale
        self.W = int(round(W * scale / 2)) * 2
        self.H = int(round(H * scale / 2)) * 2
        self.q = 4 if scale >= 0.75 else (2 if scale >= 0.4 else 1)  # glow downsample
        self.text = TextCache(scale, self.W, self.H)

    # -- data layer -----------------------------------------------------------------
    def layer(self, sc):
        """Rects, lines and dots of one scene over black: (box, rgb, alpha) with rgb the
        premultiplied colour (glow included, additive) and alpha the body coverage, both
        float32 over box = (x0, y0, x1, y1) raster px; None when nothing is drawn."""
        s, q = self.s, self.q
        rects = []
        for b in sc.rects:
            lv = b["level"]
            smx, smy = b.get("sx"), b.get("sy")      # motion-blur smear (design px), optional
            for i in np.argsort(lv, kind="stable"):      # bright ones drawn last (on top)
                if lv[i] <= 1e-3:
                    continue
                hw, hh = b["w"][i] * s / 2, b["h"][i] * s / 2
                x, y = b["x"][i] * s, b["y"][i] * s
                hx = abs(float(smx[i])) * s / 2 if smx is not None else 0.0
                hy = abs(float(smy[i])) * s / 2 if smy is not None else 0.0
                if x + hw + hx < -40 or x - hw - hx > self.W + 40 or y + hh + hy < -40 or y - hh - hy > self.H + 40:
                    continue
                rects.append((x - hw, y - hh, x + hw, y + hh, b["rgb"][i], lv[i], b["glow"][i], hx, hy,
                              b.get("fixed", False)))
        sm = getattr(sc, "line_smear", None)
        lines = [(l[0] * s, l[1] * s, l[2] * s, l[3] * s, max(l[4] * s, 0.35)) + l[5:8]
                 + ((sm[k][0] * s / 2, sm[k][1] * s / 2) if sm else (0.0, 0.0))   # signed half smear
                 for k, l in enumerate(sc.lines)]
        dots = [(d[0] * s, d[1] * s, max(d[2] * s, 0.6)) + d[3:] for d in sc.dots]
        if not (rects or lines or dots):
            return None
        sig_in = GLOW_SIGMA_IN * sc.zoom * s
        sig_out = GLOW_SIGMA_OUT * sc.zoom * s
        margin = 3.2 * sig_out + 4
        xs0 = ([r[0] - r[7] for r in rects] + [min(l[0], l[2]) - l[4] - abs(l[8]) for l in lines]
               + [d[0] - d[2] for d in dots])
        xs1 = ([r[2] + r[7] for r in rects] + [max(l[0], l[2]) + l[4] + abs(l[8]) for l in lines]
               + [d[0] + d[2] for d in dots])
        ys0 = ([r[1] - r[8] for r in rects] + [min(l[1], l[3]) - l[4] - abs(l[9]) for l in lines]
               + [d[1] - d[2] for d in dots])
        ys1 = ([r[3] + r[8] for r in rects] + [max(l[1], l[3]) + l[4] + abs(l[9]) for l in lines]
               + [d[1] + d[2] for d in dots])
        bx0 = max(0, int(math.floor(min(xs0) - margin)) // q * q)
        by0 = max(0, int(math.floor(min(ys0) - margin)) // q * q)
        bx1 = min(self.W, -(-int(math.ceil(max(xs1) + margin)) // q) * q)
        by1 = min(self.H, -(-int(math.ceil(max(ys1) + margin)) // q) * q)
        if bx1 <= bx0 or by1 <= by0:
            return None
        box = (bx0, by0, bx1, by1)
        rgb, alpha = self._composite(sc, rects, lines, dots, box, sig_in, sig_out)
        return box, rgb, alpha

    def layers_blurred(self, scenes, bounds):
        """Motion blur over the shutter interval. scenes[k] is the scene at the centre of
        sub-interval k and bounds[k], bounds[k+1] the scenes at its ends: each rect (and each
        axis-aligned line) of scenes[k] is box-blurred along its own displacement over the
        sub-interval, so the sub-frames tile the motion without ripple. Batches with a fixed
        smear (motion trails) keep theirs. Returns the sub-frame layers for compose()."""
        for k, sc in enumerate(scenes):
            A, B = bounds[k], bounds[k + 1]
            for bc, ba, bb in zip(sc.rects, A.rects, B.rects):
                if bc.get("fixed"):
                    continue
                if len(ba["x"]) == len(bb["x"]) == len(bc["x"]):
                    bc["sx"], bc["sy"] = bb["x"] - ba["x"], bb["y"] - ba["y"]
            if len(A.lines) == len(B.lines) == len(sc.lines):
                sc.line_smear = [(0.5 * (lb[0] - la[0] + lb[2] - la[2]), 0.5 * (lb[1] - la[1] + lb[3] - la[3]))
                                 for la, lb in zip(A.lines, B.lines)]
        return [self.layer(sc) for sc in scenes]

    def compose(self, frame, layers):
        """Average the data layers (sub-frames; None = nothing drawn in that one) in linear
        light and composite them over the background frame (uint8, modified in place):
        out = data + background * (1 - alpha)."""
        count = len(layers)
        layers = [L for L in layers if L is not None]
        if not layers:
            return frame
        x0 = min(L[0][0] for L in layers)
        y0 = min(L[0][1] for L in layers)
        x1 = max(L[0][2] for L in layers)
        y1 = max(L[0][3] for L in layers)
        acc = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        aacc = np.zeros((y1 - y0, x1 - x0), np.float32)
        for (bx0, by0, bx1, by1), rgb, alpha in layers:
            idx = (rgb * 255.0 + 0.5).astype(np.uint8)
            acc[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] += SRGB_TO_LIN[idx]
            aacc[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] += alpha
        if count > 1:
            acc *= 1.0 / count
            aacc *= 1.0 / count
        bg = SRGB_TO_LIN[frame[y0:y1, x0:x1]]
        bg *= (1.0 - np.clip(aacc, 0.0, 1.0))[..., None]
        acc += bg
        idx = (np.clip(acc, 0.0, 1.0) * (_LIN_N - 1) + 0.5).astype(np.int32)
        frame[y0:y1, x0:x1] = LIN_TO_SRGB[idx]
        return frame

    def _composite(self, sc, rects, lines, dots, box, sig_in, sig_out):
        q = self.q
        bx0, by0, bx1, by1 = box
        bw, bh = bx1 - bx0, by1 - by0
        rgb = np.zeros((bh, bw, 3), np.float32)
        alpha = np.zeros((bh, bw), np.float32)

        # glow source, downsampled by q, padded so the FFT blur does not wrap around
        pad = int(math.ceil(3.2 * sig_out / q)) + 2
        gh, gw = _next_fast(bh // q + 2 * pad), _next_fast(bw // q + 2 * pad)
        g = np.zeros((gh, gw, 3), np.float32)
        gx0, gy0 = bx0 / q - pad, by0 / q - pad        # glow-buffer origin in q-pixels

        for x0, y0, x1, y1, col, lv, gl, hx, hy, trail in rects:
            c = _box_cov(x0 - bx0, y0 - by0, x1 - bx0, y1 - by0, bw, bh, hx, hy)
            body = col * lv
            m = float(body.max())
            if lv > 1.0:          # brightness lifts above the nominal colour: soft, hue-kept cap
                kn = min(max(BODY_KNEE, float(col.max())), 0.999)
                if m > kn:
                    body = body * ((kn + (1 - kn) * (1 - math.exp(-(m - kn) / (1 - kn)))) / m)
            if c is not None:
                i0, i1, j0, j1, cov = c
                a = cov * min(lv, 1.0)
                if trail:      # afterimages: max-blended, so overlapping trails never pile up
                    reg = rgb[j0:j1, i0:i1]
                    np.maximum(reg, cov[..., None] * body.astype(np.float32), out=reg)
                    np.maximum(alpha[j0:j1, i0:i1], a, out=alpha[j0:j1, i0:i1])
                    continue
                reg = rgb[j0:j1, i0:i1]
                reg *= (1.0 - a)[..., None]
                reg += cov[..., None] * body.astype(np.float32)
                alpha[j0:j1, i0:i1] += (1.0 - alpha[j0:j1, i0:i1]) * a
            if gl > 0:
                c = _box_cov(x0 / q - gx0, y0 / q - gy0, x1 / q - gx0, y1 / q - gy0, gw, gh, hx / q, hy / q)
                if c is not None:
                    i0, i1, j0, j1, cov = c
                    g[j0:j1, i0:i1] += (cov * (lv * gl))[..., None] * col.astype(np.float32)

        for x0, y0, x1, y1, wd, col, al, gl, hx, hy in lines:
            self._line(rgb, alpha, g, (x0 - bx0, y0 - by0, x1 - bx0, y1 - by0), wd, col, al, bw, bh, False,
                       hx, hy)
            if gl > 0:
                self._line(None, None, g, (x0 / q - gx0, y0 / q - gy0, x1 / q - gx0, y1 / q - gy0),
                           max(wd / q, 0.35), col, al * gl, gw, gh, True, hx / q, hy / q)

        for x, y, r, col, al, gl in dots:
            c = _seg_cov(x - bx0, y - by0, x - bx0, y - by0, 2 * r, bw, bh)
            if c is not None:
                i0, i1, j0, j1, cov = c
                a = cov * al
                rgb[j0:j1, i0:i1] = np.maximum(rgb[j0:j1, i0:i1], a[..., None] * col.astype(np.float32))
                alpha[j0:j1, i0:i1] = np.maximum(alpha[j0:j1, i0:i1], a)
            if gl > 0:
                c = _seg_cov(x / q - gx0, y / q - gy0, x / q - gx0, y / q - gy0, max(2 * r / q, 0.5), gw, gh)
                if c is not None:
                    i0, i1, j0, j1, cov = c
                    g[j0:j1, i0:i1] += (cov * al * gl)[..., None] * col.astype(np.float32)

        # blur the glow source with an analytic two-gaussian kernel in the frequency domain
        fy = np.fft.fftfreq(gh)[:, None]
        fx = np.fft.rfftfreq(gw)[None, :]
        r2 = (fx * fx + fy * fy) * (2 * math.pi ** 2)
        si, so = sig_in / q, sig_out / q
        k = (GLOW_GAIN_IN * np.exp(-r2 * si * si) + GLOW_GAIN_OUT * np.exp(-r2 * so * so)) * sc.glow_gain
        G = np.fft.irfft2(np.fft.rfft2(g, axes=(0, 1)) * k[..., None].astype(np.complex64),
                          s=(gh, gw), axes=(0, 1)).astype(np.float32)
        if q > 1:
            box = (pad, pad, pad + bw / q, pad + bh / q)
            up = np.empty((bh, bw, 3), np.float32)
            for ch in range(3):
                im = Image.fromarray(np.ascontiguousarray(G[:, :, ch]), mode="F")
                up[:, :, ch] = np.asarray(im.resize((bw, bh), Image.BILINEAR, box=box))
        else:
            up = G[pad:pad + bh, pad:pad + bw]
        np.maximum(up, 0.0, out=up)
        m = up.max(axis=2)
        hot = m > GLOW_KNEE
        if hot.any():
            mh = m[hot]
            room = GLOW_CEIL - GLOW_KNEE
            up[hot] *= ((GLOW_KNEE + room * (1.0 - np.exp(-(mh - GLOW_KNEE) / room))) / mh)[:, None]
        up *= (1.0 - alpha)[..., None]
        rgb += up
        np.clip(rgb, 0.0, 1.0, out=rgb)
        return rgb, np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _line(rgb, alpha, g, seg, wd, col, al, w, h, additive, hx=0.0, hy=0.0):
        x0, y0, x1, y1 = seg
        if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:   # axis aligned -> crisp box (motion-blurred)
            hw = wd / 2.0
            if abs(x1 - x0) < 1e-6:
                c = _box_cov(x0 - hw, min(y0, y1), x0 + hw, max(y0, y1), w, h, abs(hx), abs(hy))
            else:
                c = _box_cov(min(x0, x1), y0 - hw, max(x0, x1), y0 + hw, w, h, abs(hx), abs(hy))
        else:                                             # diagonal: sampled smear along (hx, hy)
            c = _seg_cov(x0, y0, x1, y1, wd, w, h, hx, hy, 1.0 if additive else 2.2)
        if c is None:
            return
        i0, i1, j0, j1, cov = c
        a = cov * al
        if additive:
            g[j0:j1, i0:i1] += a[..., None] * col.astype(np.float32)
        else:
            rgb[j0:j1, i0:i1] = np.maximum(rgb[j0:j1, i0:i1], a[..., None] * col.astype(np.float32))
            alpha[j0:j1, i0:i1] = np.maximum(alpha[j0:j1, i0:i1], a)


# ----------------------------------------------------------------------------- text
SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc"
SERIF_INDEX = 2   # Noto Serif CJK SC
MONO_CHAIN = [
    "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf",
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

CAPTION_PX, CAPTION_Y = 48, 850
ANNOT_PX, ANNOT_BASELINE, ANNOT_TRACK, ANNOT_ALPHA = 18, 1080 - 48, 0.15, 0.45
CARD_STYLE = {"card_s04": (64, 380), "card_s22": (56, 612)}   # px size, centre y


_CARD_BOX = {}


def card_box(kind, text):
    """Ink box (x0, y0, x1, y1) of a title card in design pixels."""
    key = (kind, text)
    if key not in _CARD_BOX:
        px, cy = CARD_STYLE[kind]
        l, t, r, b = ImageFont.truetype(SERIF, px, index=SERIF_INDEX).getbbox(text, anchor="mm")
        _CARD_BOX[key] = (W / 2 + l, cy + t, W / 2 + r, cy + b)
    return _CARD_BOX[key]


def has_glyph(font, ch):
    """True unless the font would draw its .notdef box for ch."""
    if ch.isspace():
        return True
    a, b = font.getmask(ch), font.getmask("\U000FFFFD")
    return not (a.size == b.size and bytes(a) == bytes(b))


class TextCache:
    def __init__(self, scale, w, h):
        self.s, self.W, self.H = scale, w, h
        self.cache = {}
        self._serif = {}
        self.mono = [ImageFont.truetype(p, max(1, round(ANNOT_PX * scale))) for p in MONO_CHAIN]
        self.mono_cell = self.mono[0].getlength("M") + ANNOT_TRACK * ANNOT_PX * scale
        self._pick = {}

    def serif(self, px):
        px = max(1, round(px * self.s))
        if px not in self._serif:
            self._serif[px] = ImageFont.truetype(SERIF, px, index=SERIF_INDEX)
        return self._serif[px]

    def mono_font(self, ch):
        if ch not in self._pick:
            for f in self.mono:
                if has_glyph(f, ch):
                    self._pick[ch] = f
                    break
            else:
                raise ValueError(f"no font has a glyph for {ch!r} (U+{ord(ch):04X})")
        return self._pick[ch]

    def _centred(self, text, px, cy):
        f = self.serif(px)
        l, t, r, b = f.getbbox(text, anchor="mm")
        ax, ay = self.W / 2, cy * self.s
        x0, y0 = int(math.floor(ax + l)) - 2, int(math.floor(ay + t)) - 2
        im = Image.new("L", (int(r - l) + 6, int(b - t) + 6), 0)
        ImageDraw.Draw(im).text((ax - x0, ay - y0), text, font=f, fill=255, anchor="mm")
        return np.asarray(im), x0, y0, CAPTION_RGB

    def _annotation(self, text):
        s = self.s
        n = len(text)
        total = n * self.mono_cell - ANNOT_TRACK * ANNOT_PX * s
        x = self.W / 2 - total / 2
        base = ANNOT_BASELINE * s
        hgt = int(ANNOT_PX * s * 2.2) + 4
        x0, y0 = int(math.floor(x)) - 4, int(math.floor(base - ANNOT_PX * s * 1.5)) - 2
        im = Image.new("L", (int(total) + 12, hgt), 0)
        d = ImageDraw.Draw(im)
        cellw = self.mono[0].getlength("M")
        for k, ch in enumerate(text):
            if ch == " ":
                continue
            f = self.mono_font(ch)
            gx = x + k * self.mono_cell + (cellw - f.getlength(ch)) / 2 - x0
            d.text((gx, base - y0), ch, font=f, fill=255, anchor="ls")
        return np.asarray(im), x0, y0, GOLD

    def get(self, kind, text):
        key = (kind, text)
        if key not in self.cache:
            if kind == "caption":
                self.cache[key] = self._centred(text, CAPTION_PX, CAPTION_Y)
            elif kind == "annotation":
                self.cache[key] = self._annotation(text)
            else:
                px, cy = CARD_STYLE[kind]
                self.cache[key] = self._centred(text, px, cy)
            if len(self.cache) > 400:          # counters create many strings
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def draw(self, frame, kind, text, opacity):
        mask, x0, y0, col = self.get(kind, text)
        if kind == "annotation":
            opacity *= ANNOT_ALPHA
        h, w = mask.shape
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
        if fx1 <= fx0 or fy1 <= fy0:
            return
        m = mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32) * (opacity / 255.0)
        reg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        reg += (col.astype(np.float32) * 255.0 - reg) * m[..., None]
        frame[fy0:fy1, fx0:fx1] = (reg + 0.5).astype(np.uint8)
