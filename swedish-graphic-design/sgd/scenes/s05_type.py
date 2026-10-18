"""Scene 05 - Typography as Shape (01:35-02:00).

Letters built from the constructed alphabet in sgd.letters:

  1. a slow zoom out from inside a giant F: oversized letterforms cropped
     by the frame read as pure geometry until the word FORM appears
  2. red guides appear and the letters snap onto them
  3. the letters break into their geometric parts, settle into an abstract
     rhythm of squares and circles, then reassemble as FUNCTION
  4. FUNCTION rebuilds itself as SPACE, which opens up to the full width
  5. weight, scale and line breaks set a rhythm for all three words
  6. every piece gathers into a single blue square for Scene 06
"""

import math

from .. import letters as L
from ..core import *  # noqa: F401,F403

START, END = 95.0, 120.0

T_PAN = bt(38)            # 95.0    camera finds a giant F
T_Z1 = bt(38, 2)          # 96.25   zoom to the O
T_Z2 = bt(39)             # 97.5    the word, cropped by the frame
T_Z3 = bt(39, 2)          # 98.75   the whole word
T_GUIDES = bt(39, 3.5)    # 99.6875 guides
T_ALIGN = bt(40)          # 100.0   letters snap to the guides
T_DECOMP = bt(40, 3)      # 101.875 letters break apart
T_ABSTRACT = bt(41)       # 102.5   abstract rhythm
T_FUNCTION = bt(41, 3)    # 104.375 FUNCTION
T_SPACE = bt(42, 3)       # 106.875 SPACE
T_TRACK = bt(43, 2)       # 108.75  SPACE opens up
T_STACK = bt(44, 2)       # 111.25  three words in a stack
T_WEIGHT = bt(45, 2)      # 113.75  weights and sizes trade places
T_LINE = bt(46, 1)        # 115.625 one line
T_COLLAPSE = bt(47)       # 117.5   one square

FORM_H = 300.0
FORM_BASE = CY + FORM_H / 2
OFFS = (-46.0, 34.0, -22.0, 40.0)        # letters drift off the baseline
SQUARE = MOD                             # hand-over: blue square, 210 px

EVENTS = [
    (T_PAN, "swell", {"dur": 1.1}),
    (T_Z1, "low", {"note": 33}),
    (T_Z2, "low", {"note": 36}),
    (T_Z3, "low", {"note": 38}),
    (T_GUIDES, "ticks", {"n": 2, "step": 0.12}),
    (T_ALIGN + 0.05, "snap", {"pitch": 1.0}), (T_ALIGN + 0.12, "snap", {"pitch": 1.2}),
    (T_ALIGN + 0.19, "snap", {"pitch": 0.9}), (T_ALIGN + 0.26, "snap", {"pitch": 1.3}),
    (T_DECOMP, "glide", {"dur": 0.6}),
    (T_ABSTRACT, "ticks", {"n": 13, "step": BEAT / 4}),
    (T_FUNCTION, "chime", {"note": 77}),
    (T_SPACE, "chime", {"note": 81}),
    (T_TRACK, "swell", {"dur": 1.5}),
    (T_STACK, "low", {"note": 41}),
    (T_WEIGHT, "glide", {"dur": 1.2}),
    (T_LINE, "ticks", {"n": 3, "step": 0.1}),
    (T_COLLAPSE, "swell", {"dur": 1.2}),
    (T_COLLAPSE + 1.25, "thud", {}),
]


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------

def form_layout(t):
    snap = prog(t, T_ALIGN, 0.6, out_expo)
    pieces, boxes, _ = L.word("FORM", FORM_H, 0.2 * FORM_H, CX, FORM_BASE)
    out = []
    for i, pc in pieces:
        out.append((i, L.translate(pc, 0, OFFS[i] * (1 - snap))))
    return out, boxes


def camera(t, boxes):
    s = 0.2 * FORM_H
    fx0, fy0 = boxes[0][0], boxes[0][1] + OFFS[0]
    corner = (fx0 + s, fy0 + s)
    c0 = (corner[0] + 29, corner[1] + 13)
    ob = boxes[1]
    c1 = ((ob[0] + ob[2]) / 2 + 40, (ob[1] + ob[3]) / 2 + OFFS[1])
    keys = [(T_PAN, (9.0, c0[0], c0[1] - 200)),
            (T_PAN + 1.1, (9.0, c0[0], c0[1])),
            (T_Z1, (8.6, c0[0] + 4, c0[1] + 2)),
            (T_Z1 + 0.95, (4.0, c1[0], c1[1])),
            (T_Z2, (3.85, c1[0] + 6, c1[1])),
            (T_Z2 + 0.95, (1.75, CX, CY)),
            (T_Z3, (1.70, CX, CY)),
            (T_Z3 + 0.95, (1.0, CX, CY))]
    # Zoom in log space so the scale change feels even.
    lk = [(k[0], (math.log(k[1][0]), k[1][1], k[1][2])) for k in keys]
    ls, cx, cy = track(t, lk, in_out_cubic)
    return math.exp(ls), cx, cy


def abstract_targets(pieces):
    """An abstract rhythm: every piece becomes a square or a circle, spaced
    evenly along the horizontal axis in its original left-to-right order."""
    order = sorted(range(len(pieces)), key=lambda k: L.center(pieces[k][1])[0])
    n = len(order)
    out = [None] * n
    colors = [INK] * n
    step = 1500 / (n - 1)
    for rank, k in enumerate(order):
        x = CX - 750 + rank * step
        pc = pieces[k][1]
        if pc[0] == "q":
            side = 92 if rank % 2 == 0 else 46
            rot = math.pi / 4 if rank % 4 == 3 else 0.0
            out[k] = L.square(x, CY, side, rot)
        else:
            big = pc[3] > 100
            out[k] = L.disc(x, CY, 56 if big else 34)
            colors[k] = BLUE if big else RED
    return out, colors


def form_pieces_at(t):
    """FORM through alignment, decomposition and the abstract rhythm."""
    pieces, boxes = form_layout(t)
    pe = prog(t, T_DECOMP, 0.6, in_out_cubic)
    pa = prog(t, T_ABSTRACT, 1.0, in_out_quart)
    out = []
    targets, tcolors = abstract_targets(pieces)
    for k, (i, pc) in enumerate(pieces):
        if pe > 0:
            bx = boxes[i]
            lc = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
            c = L.center(pc)
            rot = (0.2 if k % 2 else -0.15) * pe
            pc = L.transform(pc, (c[0] - lc[0]) * 0.35 * pe, (c[1] - lc[1]) * 0.35 * pe, rot)
        col = INK
        if pa > 0:
            pc = L.mix(pc, targets[k], pa)
            col = lerp_color(INK, tcolors[k], pa)
        out.append((pc, col))
    return out


def word_pieces(text, h, sw, x, base, tracking=0.08, align="center"):
    pieces, _, _ = L.word(text, h, sw * h, x, base, tracking, align)
    return [(i, pc) for i, pc in pieces]


FUNCTION_ARGS = ("FUNCTION", 210.0, 0.2, CX, CY + 105)
SPACE_ARGS = ("SPACE", 280.0, 0.2, CX, CY + 140)


def morph(src, dst, p, stagger=0.0):
    """Morph a list of (piece, colour) into a list of pieces (all INK)."""
    pairs = L.match([(None, pc) for pc, _ in src], [(None, pc) for pc in dst])
    colmap = {id(pc): c for pc, c in src}
    out = []
    xs = [L.center(d[1])[0] for _, d in pairs if d is not None] or [CX]
    x0, x1 = min(xs), max(xs) + 1e-6
    for s, d in pairs:
        if d is None:
            sp = s[1]
            out.append((L.mix(sp, L.collapsed(sp), p), colmap.get(id(sp), INK)))
            continue
        dp = d[1]
        q = p
        if stagger:
            f = (L.center(dp)[0] - x0) / (x1 - x0)
            q = clamp01((p - f * stagger) / (1 - stagger))
            q = in_out_cubic(q)
        sp = s[1] if s is not None else L.collapsed(dp)
        c0 = colmap.get(id(sp), INK)
        out.append((L.mix(sp, dp, q), lerp_color(c0, INK, q)))
    return out


# --------------------------------------------------------------------------
# stack: weight, scale and line breaks
# --------------------------------------------------------------------------
STACK_A = (("FORM", 240.0, 0.045), ("FUNCTION", 120.0, 0.16), ("SPACE", 200.0, 0.27))
STACK_B = (("FORM", 150.0, 0.27), ("FUNCTION", 205.0, 0.045), ("SPACE", 120.0, 0.16))


def stack_params(spec):
    """Flush-left stack on the margin, vertically centred, 66 px leading."""
    gap = 66.0
    total = sum(h for _, h, _ in spec) + gap * (len(spec) - 1)
    y = CY - total / 2
    out = []
    for text, h, sw in spec:
        y += h
        out.append((text, h, sw, X[0], y))
        y += gap
    return out


def line_params():
    h, sw = 92.0, 0.13
    words = ("FORM", "FUNCTION", "SPACE")
    widths = [L.word(w, h, sw * h, 0, 0)[2] for w in words]
    space = 0.55 * h
    total = sum(widths) + space * 2
    x = CX - total / 2
    out = []
    for w, ww in zip(words, widths):
        out.append((w, h, sw, x, CY + h / 2))
        x += ww + space
    return out


def lerp_params(a, b, p, py=None):
    py = p if py is None else py
    return (a[0], lerp(a[1], b[1], p), lerp(a[2], b[2], p), lerp(a[3], b[3], p), lerp(a[4], b[4], py))


def stack_state(t):
    """Parameters (text, h, stroke, x_left, baseline) of the three words."""
    a = stack_params(STACK_A)
    b = stack_params(STACK_B)
    c = line_params()
    pw = prog(t, T_WEIGHT, 1.25, in_out_quart)
    # Into one line: spread sideways first, then drop onto the common baseline,
    # so the words never pass over one another.
    plx = prog(t, T_LINE, 0.85, in_out_quart)
    ply = prog(t, T_LINE + 0.5, 0.8, in_out_quart)
    out = []
    for k in range(3):
        st = lerp_params(a[k], b[k], pw)
        st = lerp_params(st, c[k], plx, ply)
        out.append(st)
    return out


def space_wide_args(t):
    tr = lerp(0.08, 0.5, prog(t, T_TRACK, 1.5, in_out_cubic))
    return tr


# --------------------------------------------------------------------------
# draw
# --------------------------------------------------------------------------

def fill_pieces(ctx, items):
    """items: list of (piece, colour); consecutive same-colour runs share a fill."""
    run = None
    for pc, col in items:
        if run is not None and col != run:
            set_color(ctx, run)
            ctx.fill()
        L.path(ctx, pc)
        run = col
    if run is not None:
        set_color(ctx, run)
        ctx.fill()


def draw_guides(ctx, t):
    a = 1 - prog(t, T_DECOMP - 0.45, 0.45, in_out_cubic)
    if t < T_GUIDES or a <= 0:
        return
    for j, y in enumerate((FORM_BASE - FORM_H, FORM_BASE)):
        p = prog(t, T_GUIDES + 0.1 * j, 0.55, in_out_cubic)
        hline(ctx, 0, W * p, y, 2, RED, a)
    pieces, boxes, _ = L.word("FORM", FORM_H, 0.2 * FORM_H, CX, FORM_BASE)
    for j, x in enumerate((boxes[0][0], boxes[-1][2])):
        p = prog(t, T_GUIDES + 0.2 + 0.1 * j, 0.55, in_out_cubic)
        vline(ctx, x, 0, H * p, 2, RED, a)


def draw(ctx, t):
    if t < T_FUNCTION:
        items = form_pieces_at(t)
        if t < T_Z3 + 0.95:
            _, boxes = form_layout(t)
            s, cx, cy = camera(t, boxes)
            ctx.save()
            ctx.translate(CX, CY)
            ctx.scale(s, s)
            ctx.translate(-cx, -cy)
            fill_pieces(ctx, items)
            ctx.restore()
        else:
            fill_pieces(ctx, items)
        draw_guides(ctx, t)
        return

    if t < T_SPACE:
        src = form_pieces_at(T_FUNCTION)
        dst = [pc for _, pc in word_pieces(*FUNCTION_ARGS)]
        fill_pieces(ctx, morph(src, dst, prog(t, T_FUNCTION, 1.4, linear), stagger=0.35))
        return

    if t < T_STACK:
        src = [(pc, INK) for _, pc in word_pieces(*FUNCTION_ARGS)]
        tr = space_wide_args(t)
        dst = [pc for _, pc in word_pieces(*SPACE_ARGS, tracking=tr)]
        fill_pieces(ctx, morph(src, dst, prog(t, T_SPACE, 1.7, linear), stagger=0.55))
        return

    # Stack: SPACE moves into line three, FORM and FUNCTION slide in.
    st = stack_state(t)
    ps = prog(t, T_STACK, 1.25, in_out_quart)
    pc_all = []
    for k, (text, h, sw, x, base) in enumerate(st):
        if k == 2:
            wide = L.word("SPACE", 280.0, 56.0, CX, CY + 140, 0.5)
            # Interpolate SPACE from its wide centred setting into the stack.
            x_wide = CX - wide[2] / 2
            hh = lerp(280.0, h, ps)
            ssw = lerp(0.2, sw, ps)
            xx = lerp(x_wide, x, ps)
            bb = lerp(CY + 140, base, ps)
            tr = lerp(0.5, 0.08, ps)
            pieces = word_pieces("SPACE", hh, ssw, xx, bb, tr, align="left")
        else:
            pin = prog(t, T_STACK + 0.25 * k, 1.2, in_out_quart)
            if pin <= 0:
                continue
            pieces = word_pieces(text, h, sw, lerp(-1500.0, x, pin), base, align="left")
        pc_all.extend(pieces)

    pcol = prog(t, T_COLLAPSE, 1.25, in_out_quart)
    if pcol <= 0:
        fill_pieces(ctx, [(pc, INK) for _, pc in pc_all])
        return
    # Each piece first flattens into a segment of one horizontal bar, then
    # the bar contracts into a single blue square.
    target = L.square(CX, CY, SQUARE)
    p1 = prog(t, T_COLLAPSE, 0.6, in_out_cubic)
    p2 = prog(t, T_COLLAPSE + 0.55, 0.75, in_out_quart)
    col = lerp_color(INK, BLUE, prog(t, T_COLLAPSE + 0.9, 0.4, in_out_sine))
    items = []
    for _, pc in pc_all:
        if pc[0] == "q":
            xs = pc[1:9:2]
            seg = ("q", min(xs), CY - 22, max(xs), CY - 22, max(xs), CY + 22, min(xs), CY + 22)
            items.append((L.mix(L.mix(pc, seg, p1), target, p2), col))
        else:
            r = pc[3] * (1 - p1)
            if r > 0.5:
                items.append((("a", pc[1], lerp(pc[2], CY, p1), r, pc[4] * (1 - p1), pc[5], pc[6]), col))
    fill_pieces(ctx, items)
