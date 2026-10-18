"""Scene 02 - The Power of Geometry (00:20-00:45).

One continuous composition that keeps rearranging basic forms:

  1. circles intersect rectangles (the overlap turns into negative space)
  2. lines divide the frame into unequal, near-golden sections
  3. the circle becomes a square which multiplies into a modular pattern
  4. the pattern condenses into a black mass that a large blue circle
     crosses, trading mass for negative space
  5. the mass drains to an outline that becomes the frame of Scene 03

Receives a black circle (r 210) at (1380, 540). Hands over the live-area
outline (120..1800 x 120..960, 2 px) to Scene 03.
"""

import math

import cairo

from ..core import *  # noqa: F401,F403

START, END = 20.0, 45.0

# 1. circle + rectangles
T_R1 = bt(8)              # 20.0
T_C1 = bt(8, 0.5)         # 20.3125
T_Y1 = bt(9)              # 22.5
T_DOT = bt(9, 2)          # 23.75
# 2. lines and sections
T_COLLAPSE = bt(10)       # 25.0
T_L1, T_L2, T_L3 = bt(10, 1), bt(10, 1.5), bt(10, 2)
T_SHIFT = bt(11, 2)       # 28.75
# 3. squares multiply
T_SQ = bt(12)             # 30.0
T_SPLIT = (bt(12, 2), bt(12, 3), bt(13), bt(13, 1))   # 31.25 31.875 32.5 33.125
T_WAVE = bt(13, 2)        # 33.75
# 4. mass and the blue circle
T_MASS = bt(14)           # 35.0
T_BC_IN = bt(14, 2)       # 36.25
T_BC_CROSS = bt(15, 2)    # 38.75
T_BC_BACK = bt(16, 2)     # 41.25
# 5. hand-over
T_DRAIN = bt(17)          # 42.5
T_OUTLINE = bt(17, 2)     # 43.75

C_START = (1380.0, 540.0, 210.0)
R1_BOX = (X[1], Y[1], X[5], Y[3])          # 330..1170 x 330..750
Y1_BOX = (X[5], Y[0], X[6], Y[1])          # 1170..1380 x 120..330
MASS = (X[4], Y[0], X[8], Y[4])            # 960..1800 x 120..960
LIVE = (X[0], Y[0], X[8], Y[4])

EVENTS = [
    (T_R1 + 0.7, "thud", {}),
    (T_C1, "glide", {"dur": 1.4}),
    (T_Y1 + 0.55, "click", {"pitch": 1.2}),
    (T_DOT, "bell", {"note": 81}),
    (T_COLLAPSE, "low", {"note": 41}),
    (T_L1, "tick", {}), (T_L2, "tick", {}), (T_L3, "tick", {}),
    (T_L3 + 0.2, "thud", {}),
    (T_SHIFT, "glide", {"dur": 1.1}),
    (T_SQ, "low", {"note": 38}),
    (T_SPLIT[0], "snap", {"pitch": 0.8}), (T_SPLIT[1], "snap", {"pitch": 1.0}),
    (T_SPLIT[2], "snap", {"pitch": 1.25}), (T_SPLIT[3], "snap", {"pitch": 1.5}),
    (T_WAVE, "ticks", {"n": 8, "step": BEAT / 4}),
    (T_MASS, "low", {"note": 33}),
    (T_BC_IN, "swell", {"dur": 1.8}),
    (T_BC_CROSS, "sub", {"note": 38}),
    (T_BC_BACK, "glide", {"dur": 1.6}),
    (T_DRAIN, "swell", {"dur": 2.2}),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def box_lerp(a, b, p):
    return tuple(lerp(a[i], b[i], p) for i in range(4))


def fill_b(ctx, b, c, a=1.0):
    fill_box(ctx, b[0], b[1], b[2], b[3], c, a)


def knockout_circle(ctx, cx, cy, r, boxes, base=INK, hole=PAPER):
    """Circle in `base`, turning into `hole` wherever it overlaps a box."""
    if r <= 0:
        return
    fill_circle(ctx, cx, cy, r, base)
    for b in boxes:
        if b[2] - b[0] <= 0 or b[3] - b[1] <= 0:
            continue
        ctx.save()
        clip_box(ctx, *b)
        fill_circle(ctx, cx, cy, r, hole)
        ctx.restore()


# --------------------------------------------------------------------------
# 1 + 2: circles, rectangles, lines
# --------------------------------------------------------------------------

def partition(t):
    """Line positions of the asymmetric partition (L1x, L2y, L3x, L4y)."""
    p = prog(t, T_SHIFT, 1.25, in_out_quart)
    return (lerp(X[3], X[2], p), lerp(Y[1], Y[2], p),
            lerp(X[6], X[5], p), lerp(Y[3], Y[2], p))


def draw_part12(ctx, t):
    l1x, l2y, l3x, l4y = partition(t)
    retract = prog(t, T_SQ, 0.6, in_out_cubic)          # step 3 starts

    # Blue rectangle: slides in, then becomes the lower-left section B.
    r1 = R1_BOX
    pin = prog(t, T_R1, 1.6, out_quart)
    r1 = (lerp(-900, r1[0], pin), r1[1], lerp(-900 + (r1[2] - r1[0]), r1[2], pin), r1[3])
    pc = prog(t, T_COLLAPSE, 1.0, in_out_quart)
    r1 = box_lerp(r1, (0, l4y, l1x, H), pc)
    r1 = (r1[0], lerp(r1[1], H, retract), r1[2], r1[3])
    if r1[3] > r1[1]:
        fill_b(ctx, r1, BLUE)

    # Yellow square grows into the tall section E.
    yb = Y1_BOX
    yb = (yb[0], lerp(-220, yb[1], prog(t, T_Y1, 0.8, out_quart)), yb[2],
          lerp(-10, yb[3], prog(t, T_Y1, 0.8, out_quart)))
    pe = prog(t, T_L3 + 0.2, 0.9, in_out_quart)
    yb = box_lerp(yb, (l3x, l2y, W, H), pe)
    if retract > 0:
        yb = (yb[0], yb[1], yb[2], lerp(yb[3], yb[1], retract))
    if t >= T_Y1 and yb[3] > yb[1]:
        fill_b(ctx, yb, YELLOW)

    # Rules: they grow out of the collapsed rectangle and divide the frame.
    if t >= T_COLLAPSE and retract < 1:
        k = 1 - retract
        p0 = prog(t, T_COLLAPSE + 0.7, 0.6, in_out_quart) * k
        hline(ctx, 0, l1x * p0, l4y, 2, INK)
        p1 = prog(t, T_L1, 0.6, in_out_quart) * k
        vline(ctx, l1x, 0, H * p1, 2, INK)
        p2 = prog(t, T_L2, 0.6, in_out_quart) * k
        hline(ctx, l1x, l1x + (W - l1x) * p2, l2y, 2, INK)
        p3 = prog(t, T_L3, 0.6, in_out_quart) * k
        vline(ctx, l3x, l2y, l2y + (H - l2y) * p3, 2, INK)

    # Circle: crosses the blue rectangle, then settles into section A.
    cx, cy, r = C_START
    p = prog(t, T_C1, 1.6, in_out_cubic)
    cx = lerp(cx, X[5], p)
    pa = prog(t, T_COLLAPSE, 1.1, in_out_quart)
    cx, cy = lerp(cx, l1x / 2, pa), lerp(cy, l4y / 2, pa)
    r = lerp(r, min(l1x, l4y) * 0.3, pa)
    if t < T_SQ:
        knockout_circle(ctx, cx, cy, r, [r1])

    # Red dot: the focal point at the centre, then at a line crossing.
    if t >= T_DOT:
        rd = 22 * prog(t, T_DOT, 0.5, out_cubic) * (1 - retract)
        dx, dy = X[5], CY
        pd = prog(t, T_COLLAPSE + 0.6, 1.0, in_out_quart)
        dx, dy = lerp(dx, l3x, pd), lerp(dy, l2y, pd)
        fill_circle(ctx, dx, dy, rd, RED)
    return cx, cy, r


# --------------------------------------------------------------------------
# 3: squares multiply
# --------------------------------------------------------------------------
LEVELS = [  # (cols, rows, cell, size)
    (1, 1, None, 420.0),
    (2, 1, 840.0, 420.0),
    (4, 2, 420.0, 210.0),
    (8, 4, 210.0, 105.0),
    (16, 8, 105.0, 52.5),
]


def level_center(k, i, j):
    if k == 0:
        return (CX, CY)
    cols, rows, cell, size = LEVELS[k]
    return (X[0] + cell * (i + 0.5), Y[0] + cell * (j + 0.5))


def level_color(k, i, j):
    """Each subdivision rasterises the same circle more finely: blue inside,
    a yellow ring on its edge, a red core once the cells are small enough."""
    if k == 0:
        return INK
    cols, rows = LEVELS[k][0], LEVELS[k][1]
    ux = (i + 0.5) * 16 / cols - 8
    uy = (j + 0.5) * 8 / rows - 4
    d = math.hypot(ux, uy)
    if d < 0.9:
        return RED
    if d < 3.0:
        return BLUE
    if d < 3.9:
        return YELLOW
    return INK


def draw_squares(ctx, t):
    """Level 0 is the circle morphing into a square; then the splits."""
    level = sum(1 for ts in T_SPLIT if t >= ts)
    if level == 0:
        return False
    cols, rows, cell, size = LEVELS[level]
    ts = T_SPLIT[level - 1]
    p = prog(t, ts, 0.5, in_out_cubic)
    pcol = prog(t, ts + 0.15, 0.45, in_out_cubic)
    mass = prog(t, T_MASS, 1.0, in_out_cubic)
    for j in range(rows):
        for i in range(cols):
            tx, ty_ = level_center(level, i, j)
            if level == 1:
                sx, sy = CX, CY
                pc = level_color(0, 0, 0)
            else:
                pi_, pj = i // 2, j // 2
                px, py = level_center(level - 1, pi_, pj)
                sx = px + (i % 2 - 0.5) * size
                sy = py + (j % 2 - 0.5) * size
                pc = level_color(level - 1, pi_, pj)
            x, y = lerp(sx, tx, p), lerp(sy, ty_, p)
            col = lerp_color(pc, level_color(level, i, j), pcol)
            s = size
            rot = 0.0
            rnd = 0.0
            if level == 4:
                # Diagonal wave: squares turn into diamonds and circles.
                w = prog(t, T_WAVE + (i + j) * 0.035, 0.45, in_out_sine) - \
                    prog(t, T_WAVE + 0.5 + (i + j) * 0.035, 0.45, in_out_sine)
                rot = w * math.pi / 4
                rnd = w if (i + j) % 2 else 0.0
                if mass > 0:
                    if i >= 8:
                        s = lerp(size, cell, prog(t, T_MASS + 0.15, 0.85, in_out_cubic))
                        col = lerp_color(col, INK, prog(t, T_MASS, 0.22, in_out_sine))
                    else:
                        s = size * (1 - prog(t, T_MASS + i * 0.03, 0.35, in_out_cubic))
            if s <= 0:
                continue
            fill_morph(ctx, x, y, s, s, rnd, col, rot=rot)
    return True


def draw_part3(ctx, t, circ):
    # The circle from section A becomes a 420 px square at the centre.
    if t < T_SPLIT[0]:
        cx, cy, r = circ
        p = prog(t, T_SQ, 0.95, in_out_cubic)
        x, y = lerp(cx, CX, p), lerp(cy, CY, p)
        s = lerp(2 * r, 420, p)
        rnd = 1 - prog(t, T_SQ + 0.3, 0.9, in_out_cubic)
        fill_morph(ctx, x, y, s, s, rnd, INK)
        return
    if t < T_MASS + 1.0:
        draw_squares(ctx, t)


# --------------------------------------------------------------------------
# 4 + 5: mass, blue circle, drain to outline
# --------------------------------------------------------------------------

def draw_part45(ctx, t):
    drain = prog(t, T_DRAIN + 0.3, 1.0, in_out_cubic)
    grow = prog(t, T_OUTLINE, 1.1, in_out_quart)
    x0 = lerp(MASS[0], LIVE[0], grow)
    box = (x0, MASS[1], MASS[2], MASS[3])
    if drain < 1:
        fill_b(ctx, box, INK)
        if drain > 0:
            inset = lerp((box[3] - box[1]) / 2, 2, drain)
            fill_box(ctx, box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset, PAPER)
    else:
        ctx.rectangle(box[0], box[1], box[2] - box[0], box[3] - box[1])
        ctx.rectangle(box[0] + 2, box[1] + 2, box[2] - box[0] - 4, box[3] - box[1] - 4)
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        set_color(ctx, INK)
        ctx.fill()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)

    # Large blue circle: enters, crosses the mass, settles half in, half out.
    r = 315.0 * (1 - prog(t, T_DRAIN, 0.8, in_out_cubic))
    if t >= T_BC_IN and r > 0:
        cx = keyed(t, [(T_BC_IN, -330.0), (T_BC_IN + 1.9, X[3] - 105),
                       (T_BC_CROSS, X[3] - 105), (T_BC_CROSS + 1.9, X[6]),
                       (T_BC_BACK, X[6]), (T_BC_BACK + 1.25, X[4])], in_out_sine)
        fill_circle(ctx, cx, CY, r, BLUE)
        ctx.save()
        clip_box(ctx, *MASS)
        fill_circle(ctx, cx, CY, r, PAPER)
        ctx.restore()


def draw(ctx, t):
    if t < T_SPLIT[0]:
        circ = draw_part12(ctx, t)
        if t >= T_SQ:
            # Step 3 starts on top of the retracting partition.
            draw_part3(ctx, t, circ)
    elif t < T_MASS + 1.0:
        draw_part3(ctx, t, None)
    else:
        draw_part45(ctx, t)
