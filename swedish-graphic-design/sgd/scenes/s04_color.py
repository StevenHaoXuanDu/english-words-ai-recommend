"""Scene 04 - Color as Communication (01:10-01:35).

A sequence of plates, each transforming into the next:

  1. the seven bands from Scene 03 merge into two large fields carrying
     contrasting circles
  2. an inverse split screen; the circles swap sides and vanish into the
     field of their own colour - colour alone makes figure and ground
  3. blue and yellow rectangles overprint; their overlap is green
  4. a quiet neutral composition in which one small red circle holds the
     eye wherever it goes
  5. the same colours in changing proportions: each change of scale
     changes the hierarchy, ending in restraint
"""

from ..core import *  # noqa: F401,F403
from .s03_grid import BANDS, BAND_COLORS

START, END = 70.0, 95.0

T_MERGE = bt(28, 1)         # 70.625 bands merge into two fields
T_CIRCLES = bt(29)          # 72.5   contrasting circles
T_SPLIT = bt(29, 3)         # 74.375 inverse split screen
T_SWAP = bt(30, 1)          # 75.625 circles swap sides
T_OVER = bt(30, 3.5)        # 77.1875 fields contract into overprinting rects
T_OVER2 = bt(31, 2.5)       # 79.0625 full overprint
T_NEUTRAL = bt(32, 1)       # 80.625 neutral composition
T_DOT = bt(32, 3.5)         # 82.1875 red circle
T_DOT2 = bt(33, 1.5)        # 83.4375
T_DOT3 = bt(33, 3.5)        # 84.6875
T_PROP = (bt(34, 2), bt(35, 0.5), bt(35, 3), bt(36, 1.5))   # 86.25 87.8125 89.375 90.9375
T_OUT = bt(37, 2)           # 93.75  clear for Scene 05

FIELD_W = (720, 1200)                     # blue | yellow, 3 : 5
REC_B = (X[1], Y[0] + HALF, X[5], Y[4] - HALF)
REC_Y = (X[3], Y[0] + HALF, X[7], Y[4] - HALF)
REC_FULL = (X[2], Y[0] + HALF, X[6], Y[4] - HALF)
N_BLOCK = (X[0], Y[0], X[3], Y[4])        # neutral composition: the black block
N_GREY = (X[4], Y[3], X[6], Y[4])
DOT_R = 21.0
DOT_P = ((X[6] + HALF, Y[2] + HALF), (X[1] + HALF, Y[2]), (X[7] + HALF, Y[0] + HALF))
# Proportion plates: widths of blue | yellow | red fields (sum = W).
PROPS = ((1440, 384, 96), (288, 1536, 96), (96, 384, 1440))
# Final restrained plate: small accents on paper.
REST_B = (X[0], Y[2], X[2], Y[4])
REST_Y = (X[2], Y[3], X[3], Y[4])
REST_DOT = (X[7] + HALF, Y[1] + HALF)

EVENTS = [
    (T_MERGE, "glide", {"dur": 1.2}),
    (T_CIRCLES, "bell", {"note": 79}),
    (T_CIRCLES + 0.3125, "bell", {"note": 74}),
    (T_SPLIT, "low", {"note": 41}),
    (T_SWAP, "swell", {"dur": 1.3}),
    (T_OVER, "thud", {}),
    (T_OVER2, "chime", {"note": 78}),
    (T_NEUTRAL, "low", {"note": 36}),
    (T_DOT, "bell", {"note": 86}),
    (T_DOT2, "click", {"pitch": 1.4}),
    (T_DOT3, "click", {"pitch": 1.6}),
    (T_PROP[0], "low", {"note": 38}),
    (T_PROP[1], "low", {"note": 41}),
    (T_PROP[2], "low", {"note": 45}),
    (T_PROP[3], "chime", {"note": 81}),
    (T_OUT, "swell", {"dur": 1.0}),
]


def box_lerp(a, b, p):
    return tuple(a[i] + (b[i] - a[i]) * p for i in range(4))


def fill_b(ctx, b, c, a=1.0):
    fill_box(ctx, b[0], b[1], b[2], b[3], c, a)


def draw_bands(ctx, widths, colors):
    x = 0.0
    for w, c in zip(widths, colors):
        if w > 0.01:
            fill_box(ctx, x, 0, x + w + 1, H, c)
        x += w


def plate12(ctx, t):
    # Bands merge into blue | yellow.
    p = prog(t, T_MERGE, 1.25, in_out_quart)
    base = [b[2] - b[0] for b in BANDS]
    target = [0, FIELD_W[0], 0, FIELD_W[1], 0, 0, 0]
    widths = [lerp(base[k], target[k], p) for k in range(7)]
    ps = prog(t, T_SPLIT, 1.0, in_out_quart)
    if ps > 0:
        widths = [lerp(widths[k], (0, 960, 0, 960, 0, 0, 0)[k], ps) for k in range(7)]
    draw_bands(ctx, widths, BAND_COLORS)
    bw = widths[0] + widths[1]
    # Contrasting circles, centred in their fields.
    ry = 150 * prog(t, T_CIRCLES, 0.7, out_cubic)
    rb = 270 * prog(t, T_CIRCLES + BEAT / 2, 0.7, out_cubic)
    ry, rb = lerp(ry, 210, ps), lerp(rb, 210, ps)
    yx = bw / 2
    bx = bw + (W - bw) / 2
    pw = prog(t, T_SWAP, 1.25, in_out_cubic)
    yx, bx = lerp(yx, 1440, pw), lerp(bx, 480, pw)
    if ry > 0:
        fill_circle(ctx, yx, CY, ry, YELLOW)
    if rb > 0:
        fill_circle(ctx, bx, CY, rb, BLUE)


def plate3(ctx, t):
    p = prog(t, T_OVER, 1.1, in_out_quart)
    b = box_lerp((0, 0, 961, H), REC_B, p)
    y = box_lerp((960, 0, W, H), REC_Y, p)
    p2 = prog(t, T_OVER2, 0.9, in_out_quart)
    b, y = box_lerp(b, REC_FULL, p2), box_lerp(y, REC_FULL, p2)
    pn = prog(t, T_NEUTRAL, 1.0, in_out_quart)
    g = (max(b[0], y[0]), max(b[1], y[1]), min(b[2], y[2]), min(b[3], y[3]))
    if pn > 0:
        # The green overprint becomes the black block of the neutral plate.
        g = box_lerp(g, N_BLOCK, pn)
        fill_b(ctx, g, lerp_color(GREEN, INK, prog(t, T_NEUTRAL, 0.35, in_out_sine)))
        return
    fill_b(ctx, b, BLUE)
    fill_b(ctx, y, YELLOW)
    if g[2] > g[0] and g[3] > g[1]:
        fill_b(ctx, g, GREEN)


def neutral_elements(ctx, t, k=1.0):
    """Thin rules and a pale grey panel: a calm, colourless composition."""
    for i, yy in enumerate((Y[1], Y[2], Y[3])):
        p = prog(t, T_NEUTRAL + 0.3 + 0.12 * i, 0.8, in_out_quart) * k
        hline(ctx, X[4], X[4] + (X[8] - X[4]) * p, yy, 2, INK)
    p = prog(t, T_NEUTRAL + 0.6, 0.8, in_out_quart) * k
    vline(ctx, X[4], Y[0], Y[0] + (Y[4] - Y[0]) * p, 2, INK)
    pg = prog(t, T_NEUTRAL + 0.7, 0.7, in_out_cubic) * k
    if pg > 0:
        fill_box(ctx, N_GREY[0], N_GREY[3] - (N_GREY[3] - N_GREY[1]) * pg, N_GREY[2], N_GREY[3], INK, 0.10)


def dot_pos(t):
    return track(t, [(T_DOT2, DOT_P[0]), (T_DOT2 + 0.9, DOT_P[1]),
                     (T_DOT3, DOT_P[1]), (T_DOT3 + 0.9, DOT_P[2])])


def plate4(ctx, t):
    plate3(ctx, t)
    k = 1 - prog(t, T_PROP[0], 0.6, in_out_cubic)
    neutral_elements(ctx, t, k)
    r = DOT_R * prog(t, T_DOT, 0.45, out_cubic)
    if r > 0:
        x, y = dot_pos(t)
        fill_circle(ctx, x, y, r, RED)


def plate5(ctx, t):
    """Blue | yellow | red in changing proportions, then restraint."""
    p0 = prog(t, T_PROP[0], 1.0, in_out_quart)
    widths = PROPS[0]
    for i in (1, 2):
        pi_ = prog(t, T_PROP[i], 0.95, in_out_quart)
        widths = tuple(lerp(widths[k], PROPS[i][k], pi_) for k in range(3))
    pr = prog(t, T_PROP[3], 1.15, in_out_quart)
    if p0 < 1:
        # Black block -> blue field, red dot -> red field, yellow wipes in.
        b = box_lerp(N_BLOCK, (0, 0, widths[0], H), p0)
        fill_b(ctx, b, lerp_color(INK, BLUE, prog(t, T_PROP[0], 0.4, in_out_sine)))
        yb = (widths[0], 0, widths[0] + widths[1], H)
        yb = (yb[0], yb[1], lerp(yb[0], yb[2], p0), yb[3])
        fill_b(ctx, yb, YELLOW)
        dx, dy = DOT_P[2]
        rb = box_lerp((dx - DOT_R, dy - DOT_R, dx + DOT_R, dy + DOT_R), (W - widths[2], 0, W, H), p0)
        fill_morph(ctx, (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2, rb[2] - rb[0], rb[3] - rb[1],
                   1 - p0, RED)
        neutral_elements(ctx, t, 1 - prog(t, T_PROP[0], 0.5, in_out_cubic))
        return
    x0, x1 = widths[0], widths[0] + widths[1]
    fields = ((0, 0, x0 + 1, H), (x0, 0, x1 + 1, H), (x1, 0, W, H))
    if pr <= 0:
        for b, c in zip(fields, (BLUE, YELLOW, RED)):
            fill_b(ctx, b, c)
        return
    # Restraint: the fields contract into small accents on paper.
    rb = box_lerp(fields[0], REST_B, pr)
    ry = box_lerp(fields[1], REST_Y, pr)
    d = 2 * DOT_R
    rr = box_lerp(fields[2], (REST_DOT[0] - d / 2, REST_DOT[1] - d / 2, REST_DOT[0] + d / 2, REST_DOT[1] + d / 2), pr)
    k = 1 - prog(t, T_OUT, 0.8, in_out_cubic)
    for b, c in ((rb, BLUE), (ry, YELLOW)):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        fill_box(ctx, cx - (b[2] - b[0]) * k / 2, cy - (b[3] - b[1]) * k / 2,
                 cx + (b[2] - b[0]) * k / 2, cy + (b[3] - b[1]) * k / 2, c)
    cx, cy = (rr[0] + rr[2]) / 2, (rr[1] + rr[3]) / 2
    fill_morph(ctx, cx, cy, (rr[2] - rr[0]) * k, (rr[3] - rr[1]) * k,
               prog(t, T_PROP[3] + 0.5, 0.7, in_out_cubic), RED)


def draw(ctx, t):
    if t < T_OVER:
        plate12(ctx, t)
    elif t < T_PROP[0]:
        plate4(ctx, t) if t >= T_NEUTRAL else plate3(ctx, t)
    else:
        plate5(ctx, t)
