"""Shared constants, palette, grid, easing and drawing primitives."""

import math

import cairo

# --------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------
W, H = 1920, 1080
FPS = 30
DURATION = 180.0
NFRAMES = int(DURATION * FPS)  # 5400

# --------------------------------------------------------------------------
# Grid: 120 px margins around a live area of 8 x 4 modules of 210 px.
# The 4-column grid uses 420 px modules, the 8-column grid 210 px and the
# fine grid 105 px. The frame centre (960, 540) sits on a grid crossing.
# --------------------------------------------------------------------------
M = 120
MOD = 210
HALF = MOD // 2
X = [M + MOD * i for i in range(9)]   # 120 ... 1800
Y = [M + MOD * j for j in range(5)]   # 120 ... 960
CX, CY = W / 2, H / 2

# --------------------------------------------------------------------------
# Tempo: 96 BPM in 4/4, one bar = 2.5 s = 75 frames. Every scene boundary
# in the brief (20, 45, 70, 95, 120, 145, 165 s) falls on a bar line.
# --------------------------------------------------------------------------
BPM = 96
BEAT = 60.0 / BPM
BAR = 4 * BEAT


def bt(bar, beat=0.0):
    """Seconds at a (0-indexed) bar plus beat offset."""
    return (bar * 4 + beat) * BEAT


# --------------------------------------------------------------------------
# Palette (sRGB)
# --------------------------------------------------------------------------
def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


PAPER = hexrgb("#F1ECE3")   # warm off-white
INK = hexrgb("#1F2022")     # charcoal black
BLUE = hexrgb("#3B5E7F")    # muted Nordic blue
GREEN = hexrgb("#2E5341")   # forest green
YELLOW = hexrgb("#EFD48A")  # pale yellow
RED = hexrgb("#D13F2E")     # vivid red, used sparingly

# --------------------------------------------------------------------------
# Easing. Everything is smooth and controlled: no bounce, no elastic.
# --------------------------------------------------------------------------


def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def linear(x):
    return x


def in_out_sine(x):
    return 0.5 - 0.5 * math.cos(math.pi * x)


def in_cubic(x):
    return x * x * x


def out_cubic(x):
    return 1 - (1 - x) ** 3


def in_out_cubic(x):
    return 4 * x * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def in_quart(x):
    return x ** 4


def out_quart(x):
    return 1 - (1 - x) ** 4


def in_out_quart(x):
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


def out_quint(x):
    return 1 - (1 - x) ** 5


def in_out_quint(x):
    return 16 * x ** 5 if x < 0.5 else 1 - (-2 * x + 2) ** 5 / 2


def out_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def in_expo(x):
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def in_out_expo(x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def prog(t, t0, dur, ease=in_out_cubic):
    """Eased 0..1 progress of an animation starting at t0 lasting dur."""
    if dur <= 0:
        return 1.0 if t >= t0 else 0.0
    return ease(clamp01((t - t0) / dur))


def lerp(a, b, p):
    return a + (b - a) * p


def lerp_pt(a, b, p):
    return (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)


def lerp_color(a, b, p):
    return tuple(a[i] + (b[i] - a[i]) * p for i in range(3))


def keyed(t, keys, ease=in_out_cubic):
    """Piecewise animation through keyframes [(time, value), ...].

    Values may be numbers or tuples. Each segment is eased independently,
    so every move has a clear start and end position.
    """
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t < t1:
            p = ease(clamp01((t - t0) / (t1 - t0))) if t1 > t0 else 1.0
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * p for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * p
    return keys[-1][1]


def stagger(t, t0, i, step, dur, ease=in_out_cubic):
    return prog(t, t0 + i * step, dur, ease)


def track(t, keys, ease=in_out_cubic):
    """Hold-and-move keyframes: [(time, value[, ease]), ...].

    The value holds between keys and moves from key k-1 to key k during
    their time span, shaped by key k's ease (or the default). Values may be
    numbers or tuples of numbers.
    """
    k0 = keys[0]
    if t <= k0[0]:
        return k0[1]
    for prev, key in zip(keys, keys[1:]):
        if t < key[0]:
            e = key[2] if len(key) > 2 else ease
            span = key[0] - prev[0]
            p = e(clamp01((t - prev[0]) / span)) if span > 0 else 1.0
            a, b = prev[1], key[1]
            if isinstance(a, tuple):
                return tuple(x + (y - x) * p for x, y in zip(a, b))
            return a + (b - a) * p
    return keys[-1][1]


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------


def set_color(ctx, c, a=1.0):
    if a >= 1.0:
        ctx.set_source_rgb(*c)
    else:
        ctx.set_source_rgba(c[0], c[1], c[2], a)


def fill_rect(ctx, x, y, w, h, c, a=1.0):
    if w == 0 or h == 0 or a <= 0:
        return
    ctx.rectangle(x, y, w, h)
    set_color(ctx, c, a)
    ctx.fill()


def fill_box(ctx, x0, y0, x1, y1, c, a=1.0):
    fill_rect(ctx, min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0), c, a)


def fill_circle(ctx, cx, cy, r, c, a=1.0):
    if r <= 0 or a <= 0:
        return
    ctx.new_sub_path()
    ctx.arc(cx, cy, r, 0, 2 * math.pi)
    set_color(ctx, c, a)
    ctx.fill()


def circle_path(ctx, cx, cy, r):
    ctx.new_sub_path()
    ctx.arc(cx, cy, max(r, 0.0), 0, 2 * math.pi)
    ctx.close_path()


def rrect_path(ctx, x, y, w, h, r):
    """Rounded rectangle. r = 0 gives a square corner, r = min(w, h) / 2
    gives a circle (w == h) or a stadium. Used to morph squares, rectangles
    and circles into one another."""
    r = max(0.0, min(r, abs(w) / 2, abs(h) / 2))
    if r < 0.01:
        ctx.rectangle(x, y, w, h)
        return
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


def fill_rrect(ctx, x, y, w, h, r, c, a=1.0):
    if w <= 0 or h <= 0 or a <= 0:
        return
    rrect_path(ctx, x, y, w, h, r)
    set_color(ctx, c, a)
    ctx.fill()


def morph_shape(ctx, cx, cy, w, h, roundness, rot=0.0):
    """Path for a shape centred on (cx, cy) that is a rectangle when
    roundness = 0 and fully rounded (circle / stadium) when roundness = 1."""
    ctx.save()
    ctx.translate(cx, cy)
    if rot:
        ctx.rotate(rot)
    rrect_path(ctx, -w / 2, -h / 2, w, h, roundness * min(w, h) / 2)
    ctx.restore()


def fill_morph(ctx, cx, cy, w, h, roundness, c, a=1.0, rot=0.0):
    if w <= 0 or h <= 0 or a <= 0:
        return
    morph_shape(ctx, cx, cy, w, h, roundness, rot)
    set_color(ctx, c, a)
    ctx.fill()


def line(ctx, x0, y0, x1, y1, width, c, a=1.0):
    if a <= 0 or width <= 0 or (x0 == x1 and y0 == y1):
        return
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y1)
    ctx.set_line_width(width)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    set_color(ctx, c, a)
    ctx.stroke()


def hline(ctx, x0, x1, y, width, c, a=1.0):
    """Horizontal rule as a filled rectangle centred on y (pixel crisp)."""
    if x1 < x0:
        x0, x1 = x1, x0
    fill_rect(ctx, x0, y - width / 2, x1 - x0, width, c, a)


def vline(ctx, x, y0, y1, width, c, a=1.0):
    if y1 < y0:
        y0, y1 = y1, y0
    fill_rect(ctx, x - width / 2, y0, width, y1 - y0, c, a)


def sector_path(ctx, cx, cy, r, a0, a1):
    """Pie slice from angle a0 to a1 (radians, screen orientation)."""
    ctx.move_to(cx, cy)
    ctx.arc(cx, cy, r, a0, a1)
    ctx.close_path()


def fill_sector(ctx, cx, cy, r, a0, a1, c, a=1.0):
    if r <= 0 or a <= 0:
        return
    sector_path(ctx, cx, cy, r, a0, a1)
    set_color(ctx, c, a)
    ctx.fill()


def ring_seg_path(ctx, cx, cy, r_out, r_in, a0, a1):
    """Annular segment between radii r_in and r_out, angles a0 -> a1."""
    ctx.new_sub_path()
    ctx.arc(cx, cy, r_out, a0, a1)
    if r_in > 0:
        ctx.arc_negative(cx, cy, r_in, a1, a0)
    else:
        ctx.line_to(cx, cy)
    ctx.close_path()


def clip_box(ctx, x0, y0, x1, y1):
    ctx.rectangle(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
    ctx.clip()


def polygon_path(ctx, pts):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def fill_polygon(ctx, pts, c, a=1.0):
    if a <= 0:
        return
    polygon_path(ctx, pts)
    set_color(ctx, c, a)
    ctx.fill()


def dashed_line(ctx, x0, y0, x1, y1, width, c, dash=12, gap=10, a=1.0, offset=0.0):
    ctx.save()
    ctx.set_dash([dash, gap], offset)
    line(ctx, x0, y0, x1, y1, width, c, a)
    ctx.restore()


def wipe_rect(x, y, w, h, p, direction):
    """Rectangle (x, y, w, h) revealed to fraction p from one side."""
    if direction == "right":     # grows from the left edge towards the right
        return x, y, w * p, h
    if direction == "left":
        return x + w * (1 - p), y, w * p, h
    if direction == "down":
        return x, y, w, h * p
    if direction == "up":
        return x, y + h * (1 - p), w, h * p
    if direction == "h":         # from the centre outwards horizontally
        return x + w * (1 - p) / 2, y, w * p, h
    if direction == "v":
        return x, y + h * (1 - p) / 2, w, h * p
    raise ValueError(direction)
