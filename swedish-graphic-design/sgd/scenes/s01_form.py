"""Scene 01 - The Beginning of Form (00:00-00:20).

An empty sheet. A black circle, a thin line, a blue rectangle and a small
yellow square arrive one by one, find their places on an invisible grid
and transform: the circle sinks into a semicircle, the line becomes a
divider, the rectangle expands into a colour field and the title is
revealed by geometric wipes.

Hands over to Scene 02 with a lone black circle (r 210) at (1380, 540).
"""

from .. import typography as ty
from ..core import *  # noqa: F401,F403

START, END = 0.0, 20.0

T_CIRCLE = bt(0, 2)      # 1.25  circle appears
T_LINE = bt(1, 1)        # 3.125 line extends from its edge
T_RECT = bt(2)           # 5.0   rectangle enters from the left
T_SQUARE = bt(2, 2)      # 6.25  small yellow square follows
T_GRID = bt(3)           # 7.5   the grid becomes faintly visible
T_SNAP = bt(3, 2)        # 8.75  everything aligns to the grid
T_EXPAND = bt(4)         # 10.0  rectangle expands into a colour field
T_SINK = bt(4, 2)        # 11.25 circle sinks into a semicircle
T_TITLE = bt(5)          # 12.5  title revealed
T_CAPTION = bt(6, 1)     # 15.625 caption
T_OUT = bt(7, 2)         # 18.75 hand-over begins

DIVIDER_Y = Y[3]         # 750
FIELD_X1 = X[4]          # 960
CIRCLE_END = (X[6], Y[2], MOD)   # (1380, 540, 210)

TITLE = ("SWEDISH", "GRAPHIC", "DESIGN")
TITLE_SIZE = 140
TITLE_PITCH = 134
TITLE_BASES = tuple(Y[4] - TITLE_PITCH * (2 - i) for i in range(3))   # last line on the margin
CAPTION = "A STUDY IN FORM"

EVENTS = [
    (T_CIRCLE, "bloom", {"note": 50}),
    (T_LINE, "glide", {"dur": 1.5}),
    (T_RECT + 0.9, "thud", {}),
    (T_SQUARE + 0.8, "click", {"pitch": 1.0}),
    (T_GRID, "ticks", {"n": 13, "step": 0.06}),
    (T_SNAP + 0.55, "snap", {}),
    (T_SNAP + 0.75, "snap", {}),
    (T_SNAP + 0.95, "snap", {}),
    (T_SNAP + 1.15, "snap", {}),
    (T_EXPAND, "low", {"note": 34}),
    (T_SINK, "sub", {"note": 38}),
    (T_TITLE, "chime", {"note": 74}),
    (T_TITLE + BEAT / 2, "chime", {"note": 77}),
    (T_TITLE + BEAT, "chime", {"note": 81}),
    (T_CAPTION, "tick", {}),
    (T_OUT + 0.25, "swell", {"dur": 1.2}),
]


def circle_state(t):
    r = 105 * prog(t, T_CIRCLE, 1.8, out_quint)
    cx, cy = CX, CY
    p = prog(t, T_SNAP, 1.3, in_out_quart)
    cx = lerp(cx, CIRCLE_END[0], p)
    r = lerp(r, CIRCLE_END[2], p)
    cy = lerp(cy, DIVIDER_Y, prog(t, T_SINK, 1.4, in_out_cubic))
    cy = lerp(cy, CIRCLE_END[1], prog(t, T_OUT, 0.95, in_out_cubic))
    return cx, cy, r


def line_state(t):
    x0 = CX + 105
    x1 = lerp(x0, X[7], prog(t, T_LINE, 1.6, in_out_cubic))
    y = CY
    p = prog(t, T_SNAP + 0.1, 1.3, in_out_quart)
    x0, x1, y = lerp(x0, X[4], p), lerp(x1, X[8], p), lerp(y, DIVIDER_Y, p)
    # Retract into the circle's centre once the circle has risen.
    q = prog(t, T_OUT + 0.7, 0.55, in_out_cubic)
    x0, x1 = lerp(x0, CIRCLE_END[0], q), lerp(x1, CIRCLE_END[0], q)
    return x0, x1, y


def rect_state(t):
    w, h = 2 * MOD, MOD
    x = lerp(-w - 40, 268, prog(t, T_RECT, 1.6, out_quart))
    y = 772.0
    p = prog(t, T_SNAP + 0.2, 1.3, in_out_quart)
    x, y = lerp(x, X[0], p), lerp(y, Y[3], p)
    x0, y0, x1, y1 = x, y, x + w, y + h
    pv = prog(t, T_EXPAND, 0.9, in_out_cubic)
    y0, y1 = lerp(y0, 0, pv), lerp(y1, H, pv)
    ph = prog(t, T_EXPAND + 0.7, 1.0, in_out_cubic)
    x0, x1 = lerp(x0, 0, ph), lerp(x1, FIELD_X1, ph)
    x1 = lerp(x1, 0, prog(t, T_OUT + 0.2, 1.0, in_out_cubic))
    return x0, y0, x1, y1


def square_state(t):
    s = HALF
    x = lerp(-180, 708, prog(t, T_SQUARE, 1.4, out_quart))
    y = 645.0
    p = prog(t, T_SNAP + 0.3, 1.45, in_out_quart)
    return lerp(x, X[0], p), lerp(y, Y[0], p), s


def draw_guides(ctx, t):
    fade = 1 - prog(t, T_SNAP + 1.7, 0.8, in_out_cubic)
    if t < T_GRID or fade <= 0:
        return
    a = 0.20 * fade
    span_y = Y[4] - Y[0]
    span_x = X[8] - X[0]
    for i, x in enumerate(X):
        p = prog(t, T_GRID + 0.05 * i, 0.8, in_out_cubic)
        fill_rect(ctx, x, Y[0], 1, span_y * p, INK, a)
    for j, y in enumerate(Y):
        p = prog(t, T_GRID + 0.25 + 0.05 * j, 0.8, in_out_cubic)
        fill_rect(ctx, X[0], y, span_x * p, 1, INK, a)
    # Small registration crosses where the lines meet.
    pc = prog(t, T_GRID + 0.6, 0.5, in_out_cubic)
    for x in X:
        for y in Y:
            fill_rect(ctx, x - 6, y, 13, 1, INK, a * 1.6 * pc)
            fill_rect(ctx, x, y - 6, 1, 13, INK, a * 1.6 * pc)


def draw_title(ctx, t, field_x1):
    if t < T_TITLE:
        return
    ctx.save()
    clip_box(ctx, 0, 0, field_x1, H)
    for i, word in enumerate(TITLE):
        tl = ty.line(ty.DISPLAY_BOLD, word, TITLE_SIZE, -0.012)
        base = TITLE_BASES[i]
        ox = tl.x_for(X[0], "left")
        x0, x1 = X[0], X[0] + tl.ink_width
        ts = T_TITLE + i * BEAT / 2
        p1 = prog(t, ts, 0.55, in_out_cubic)
        p2 = prog(t, ts + 0.5, 0.65, in_out_cubic)
        bar_l, bar_r = lerp(x0, x1, p2), lerp(x0, x1, p1)
        top, bot = base - tl.cap - 10, base + 10
        if p2 > 0:
            ctx.save()
            clip_box(ctx, x0 - 30, top - 20, bar_l, bot + 20)
            tl.draw(ctx, ox, base, PAPER)
            ctx.restore()
        if bar_r > bar_l:
            fill_box(ctx, bar_l, top, bar_r, bot, YELLOW)
    ctx.restore()


def draw_caption(ctx, t):
    if t < T_CAPTION:
        return
    tl = ty.line(ty.TEXT_MEDIUM, CAPTION, 24, 0.22)
    p = prog(t, T_CAPTION, 0.9, out_quart) * (1 - prog(t, T_OUT - 0.1, 0.5, in_out_cubic))
    base = lerp(DIVIDER_Y + 2, DIVIDER_Y + 62, p)
    ctx.save()
    clip_box(ctx, X[5] - 10, DIVIDER_Y + 2, X[8], DIVIDER_Y + 100)
    tl.draw(ctx, tl.x_for(X[5], "left"), base, INK)
    ctx.restore()


def draw(ctx, t):
    draw_guides(ctx, t)

    fx0, fy0, fx1, fy1 = rect_state(t)
    if t >= T_RECT and fx1 > fx0:
        fill_box(ctx, fx0, fy0, fx1, fy1, BLUE)

    draw_title(ctx, t, fx1)

    if t >= T_SQUARE:
        sx, sy, s = square_state(t)
        ctx.save()
        if t > T_OUT:
            clip_box(ctx, 0, 0, fx1, H)
        fill_rect(ctx, sx, sy, s, s, YELLOW)
        ctx.restore()

    if t >= T_LINE:
        x0, x1, y = line_state(t)
        if x1 - x0 > 0.5:
            hline(ctx, x0, x1, y, 2, INK)

    cx, cy, r = circle_state(t)
    if r > 0:
        ctx.save()
        clip_box(ctx, 0, 0, W, DIVIDER_Y)
        fill_circle(ctx, cx, cy, r, INK)
        ctx.restore()

    draw_caption(ctx, t)
