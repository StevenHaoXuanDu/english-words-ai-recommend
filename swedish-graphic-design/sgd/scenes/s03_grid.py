"""Scene 03 - Grid and Order (00:45-01:10).

A modular grid is drawn inside the outline inherited from Scene 02. Each
word appears as a graphic element and then becomes the grid:

  ORDER    the letters fill into coloured blocks that snap into cells
  BALANCE  set on the grid lines; its seven letters collapse into the seven
           rules that turn four columns into eight
  CLARITY  heads an editorial layout, then its letters become the colour
           bands that open Scene 04

Hands over seven full-frame vertical colour bands (BANDS) to Scene 04.
"""

from .. import typography as ty
from ..core import *  # noqa: F401,F403

START, END = 45.0, 70.0

T_GRID = bt(18)              # 45.0    internal lines draw
T_ORDER = bt(18, 3.5)        # 47.1875 ORDER rises
T_ORDER_FILL = bt(19, 3)     # 49.375  letters fill into blocks
T_ORDER_SNAP = bt(20)        # 50.0    blocks snap into cells
T_ASYM = bt(21)              # 52.5    blocks move outside the grid
T_BAL = bt(21, 3)            # 54.375  BALANCE opens from the midline
T_BAL_LINES = bt(22, 2)      # 56.25   letters collapse into rules
T_ROWS = bt(23)              # 57.5    rows complete the 8 x 4 grid
T_LAYOUT = bt(23, 1)         # 58.125  blocks reflow into a layout
T_CLARITY = bt(24)           # 60.0    headline
T_TEXT = bt(24, 1)           # 60.625  text lines
T_CL_FILL = bt(25, 2)        # 63.75   CLARITY fills into blocks
T_CLEAR = bt(25, 3)          # 64.375  layout clears
T_CL_SNAP = bt(26, 1)        # 65.625  blocks snap into columns
T_BANDS = bt(27)             # 67.5    columns expand into bands

LIVE = (X[0], Y[0], X[8], Y[4])
G = 15                       # half gutter used by the editorial layout

# ORDER ---------------------------------------------------------------------
ORDER_FONT = ty.DISPLAY_BLACK
ORDER_SIZE = 460.0
ORDER_COLORS = (BLUE, YELLOW, INK, GREEN, RED)
ORDER_CELLS = (
    (X[0], Y[0], X[2], Y[2]),                # O  full cell
    (X[2], Y[2], X[3], Y[3]),                # R  half
    (X[4], Y[2], X[6], Y[4]),                # D  full cell
    (X[7], Y[1], X[8], Y[2]),                # E  half
    (X[2] + HALF, Y[1] + HALF, X[3], Y[2]),  # R  quarter, resting on the half
)
ASYM = {0: (X[0] - HALF, Y[0] - HALF, X[2] - HALF, Y[2] - HALF),
        2: (X[4] + HALF, Y[2] + HALF, X[6] + HALF, Y[4] + HALF),
        3: (X[8], Y[1], X[8] + MOD, Y[2])}
IMAGE = (X[0], Y[0], X[4] - G, Y[3] - G)
SUN_C, SUN_D = ((IMAGE[0] + IMAGE[2]) / 2, (IMAGE[1] + IMAGE[3]) / 2), 300.0
QUOTE = (X[6] + G, Y[3] + G, X[8], Y[4] - G)
BULLET = (X[0], Y[3] + 22, X[0] + 22, Y[3] + 44)

# BALANCE -------------------------------------------------------------------
BAL_FONT = ty.DISPLAY_BOLD
BAL_SIZE = 206.0
BAL_X = tuple(X[1] + MOD * k for k in range(7))       # 330 ... 1590

# CLARITY -------------------------------------------------------------------
CL_FONT = ty.DISPLAY_BOLD
CL_SIZE = 196.6
CL_X = X[4] + G
CL_BASE = Y[1] - G
BAND_COLORS = (INK, BLUE, GREEN, YELLOW, RED, BLUE, INK)
BANDS = tuple((round(W * k / 7), 0, round(W * (k + 1) / 7), H) for k in range(7))


def _text_lines():
    """Abstract text: two columns of lines with paragraph breaks."""
    cols = ((X[4] + G, X[6] - G, (1, 1, 1, 1, .94, .52, 0, 1, 1, 1, 1, .97, .6, 0, 1, .34)),
            (X[6] + G, X[8], (1, 1, 1, .9, 0, 1, 1, 1, 1, 1, .44, 0, 1, 1, .98, .7)))
    out = []
    for x0, x1, widths in cols:
        for m, wf in enumerate(widths):
            if wf:
                y = Y[1] + 48 + 24 * m
                out.append((x0, y, x0 + (x1 - x0) * wf, y + 7))
    return out


TEXT = _text_lines()
CAPTION_BARS = tuple((X[0] + 45, Y[3] + 26 + 24 * i, X[0] + 45 + w, Y[3] + 33 + 24 * i)
                     for i, w in enumerate((700, 610, 380)))
QUOTE_BARS = tuple((QUOTE[0] + 30, QUOTE[1] + 38 + 26 * i, QUOTE[0] + 30 + w, QUOTE[1] + 47 + 26 * i)
                   for i, w in enumerate((340, 318, 345, 170)))

EVENTS = (
    [(T_GRID + i * BEAT / 2, "tick", {}) for i in range(3)]
    + [(T_GRID + 1.5 * BEAT, "tick", {"pitch": 1.3})]
    + [(T_ORDER, "swell", {"dur": 0.9}),
       (T_ORDER_FILL, "ticks", {"n": 5, "step": 0.07})]
    + [(T_ORDER_SNAP + k * BEAT / 2 + 0.12, "snap", {"pitch": 1 + 0.12 * k}) for k in range(5)]
    + [(T_ASYM + k * BEAT / 2, "thud", {}) for k in range(3)]
    + [(T_BAL, "chime", {"note": 76}),
       (T_BAL_LINES, "glide", {"dur": 1.0}),
       (T_ROWS, "ticks", {"n": 2, "step": 0.15}),
       (T_LAYOUT, "low", {"note": 41}),
       (T_CLARITY, "chime", {"note": 79}),
       (T_TEXT, "ticks", {"n": 10, "step": 0.05}),
       (T_CL_FILL, "ticks", {"n": 7, "step": 0.06}),
       (T_CLEAR, "swell", {"dur": 1.0})]
    + [(T_CL_SNAP + k * BEAT / 4 + 0.1, "snap", {"pitch": 0.9 + 0.08 * k}) for k in range(7)]
    + [(T_BANDS, "low", {"note": 38})]
)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def grid_alpha(t):
    return (1 - 0.84 * prog(t, T_LAYOUT, 1.2, in_out_cubic)) * (1 - prog(t, T_CLEAR, 0.8, in_out_cubic))


def bal_line_progress(t, k):
    return prog(t, T_BAL_LINES + abs(k - 3) * BEAT / 4, 0.75, in_out_quart)


def draw_grid(ctx, t, a):
    if a <= 0:
        return
    x0, y0, x1, y1 = LIVE
    hline(ctx, x0 - 1, x1 + 1, y0, 2, INK, a)
    hline(ctx, x0 - 1, x1 + 1, y1, 2, INK, a)
    vline(ctx, x0, y0, y1, 2, INK, a)
    vline(ctx, x1, y0, y1, 2, INK, a)
    for i, x in enumerate((X[2], X[4], X[6])):
        p = prog(t, T_GRID + i * BEAT / 2, 0.9, in_out_quart)
        vline(ctx, x, y0, y0 + (y1 - y0) * p, 2, INK, a)
    p = prog(t, T_GRID + 1.5 * BEAT, 0.9, in_out_quart)
    hline(ctx, x0, x0 + (x1 - x0) * p, Y[2], 2, INK, a)
    for k in (0, 2, 4, 6):
        if bal_line_progress(t, k) >= 1:
            vline(ctx, BAL_X[k], y0, y1, 2, INK, a)
    for j, y in enumerate((Y[1], Y[3])):
        p = prog(t, T_ROWS + 0.15 * j, 0.9, in_out_quart)
        hline(ctx, x0, x0 + (x1 - x0) * p, y, 2, INK, a)


# --------------------------------------------------------------------------
# words
# --------------------------------------------------------------------------

def order_line():
    tl = ty.line(ORDER_FONT, "ORDER", ORDER_SIZE)
    return tl, tl.x_for(CX, "center"), Y[1] + tl.cap / 2


def glyph_box(tl, i, ox, base):
    b = tl.glyphs[i].bounds
    return (ox + b[0], base + b[1], ox + b[2], base + b[3])


def fill_progress(t, t0, i, step=0.07):
    return prog(t, t0 + i * step, 0.35, in_out_cubic)


def draw_filling_word(ctx, t, tl, ox, base, t_fill, colors, rise=None, step=0.07):
    """Draw letters that fill, bottom to top, into solid blocks."""
    for i in range(len(tl.glyphs)):
        pf = fill_progress(t, t_fill, i, step)
        if pf >= 1:
            continue
        dy = rise(i) if rise else 0.0
        bx0, by0, bx1, by1 = glyph_box(tl, i, ox, base + dy)
        tl.glyph_path(ctx, i, ox, base + dy)
        set_color(ctx, INK)
        ctx.fill()
        if pf > 0:
            fill_box(ctx, bx0, by1 - (by1 - by0) * pf, bx1, by1, colors[i])


def order_block(t, k, start_box):
    keys = [(T_ORDER_SNAP + k * BEAT / 2, start_box),
            (T_ORDER_SNAP + k * BEAT / 2 + 0.6, ORDER_CELLS[k], out_expo)]
    if k in ASYM:
        j = (0, None, 1, 2)[k]
        ta = T_ASYM + j * BEAT / 2
        keys += [(ta, ORDER_CELLS[k]), (ta + 0.7, ASYM[k], out_expo)]
    return track(t, keys)


def draw_order(ctx, t):
    if t < T_ORDER:
        return
    tl, ox, base = order_line()
    ctx.save()
    clip_box(ctx, X[0], 0, X[8], Y[2])

    def rise(i):
        return (1 - prog(t, T_ORDER + i * 0.07, 0.75, out_quart)) * (tl.cap + 60)

    draw_filling_word(ctx, t, tl, ox, base, T_ORDER_FILL, ORDER_COLORS, rise)
    ctx.restore()
    for k in range(5):
        if fill_progress(t, T_ORDER_FILL, k) < 1 or t >= T_LAYOUT:
            continue
        b = order_block(t, k, glyph_box(tl, k, ox, base))
        fill_box(ctx, *b, ORDER_COLORS[k])


def draw_balance(ctx, t):
    if t < T_BAL or t >= T_BAL_LINES + 2:
        return
    tl = ty.line(BAL_FONT, "BALANCE", BAL_SIZE)
    base = Y[2] + tl.cap / 2
    for k, g in enumerate(tl.glyphs):
        cx = BAL_X[k]
        gx = cx - (g.bounds[0] + g.bounds[2]) / 2       # line origin for this glyph
        po = prog(t, T_BAL + k * 0.06, 0.6, in_out_cubic)
        pl = bal_line_progress(t, k)
        if pl > 0:
            half = lerp(tl.cap / 2, (Y[4] - Y[0]) / 2, pl)
            vline(ctx, cx, Y[2] - half, Y[2] + half, 2, INK)
        sx = 1 - pl
        if po <= 0 or sx <= 0.002:
            continue
        ctx.save()
        h = (tl.cap / 2 + 8) * po
        clip_box(ctx, 0, Y[2] - h, W, Y[2] + h)
        ctx.translate(cx, 0)
        ctx.scale(sx, 1)
        ctx.translate(-cx, 0)
        tl.glyph_path(ctx, k, gx, base)
        set_color(ctx, INK)
        ctx.fill()
        ctx.restore()


# --------------------------------------------------------------------------
# editorial layout
# --------------------------------------------------------------------------

def layout_p(t, k):
    return prog(t, T_LAYOUT + k * BEAT / 4, 1.15, in_out_quart)


def clear_p(t, d=0.0, dur=0.6):
    return prog(t, T_CLEAR + d, dur, in_out_cubic)


def draw_bar(ctx, box, p, c, a=1.0):
    """A bar that grows from its left end (p = 0..1)."""
    if p <= 0:
        return
    fill_box(ctx, box[0], box[1], box[0] + (box[2] - box[0]) * p, box[3], c, a)


def draw_layout(ctx, t):
    if t < T_LAYOUT:
        return
    # Image block (from the blue O), with the yellow R turned into a sun.
    img = tuple(lerp(ASYM[0][i], IMAGE[i], layout_p(t, 0)) for i in range(4))
    pc = clear_p(t, 0.1, 0.7)
    img = (img[0], img[1], img[2], lerp(img[3], img[1], pc))
    if img[3] > img[1]:
        fill_box(ctx, *img, BLUE)
    p = layout_p(t, 1)
    y0 = ORDER_CELLS[1]
    cx = lerp((y0[0] + y0[2]) / 2, SUN_C[0], p)
    cy = lerp((y0[1] + y0[3]) / 2, SUN_C[1], p)
    d = lerp(MOD, SUN_D, p) * (1 - clear_p(t, 0.0, 0.5))
    if d > 0:
        ctx.save()
        if pc > 0:
            clip_box(ctx, *img)
        fill_morph(ctx, cx, cy, d, d, prog(t, T_LAYOUT + 0.3, 0.9, in_out_cubic), YELLOW)
        ctx.restore()

    # The black D slices into the lines of two text columns.
    src = ASYM[2]
    n = len(TEXT)
    for i, dst in enumerate(TEXT):
        p = prog(t, T_LAYOUT + 0.25 + 0.018 * i, 1.0, in_out_quart)
        s = (src[0], src[1] + (src[3] - src[1]) * i / n, src[2], src[1] + (src[3] - src[1]) * (i + 1) / n)
        box = tuple(lerp(s[j], dst[j], p) for j in range(4))
        k = 1 - clear_p(t, 0.004 * i, 0.5)
        if k > 0:
            fill_box(ctx, box[0], box[1], box[0] + (box[2] - box[0]) * k, box[3], INK)

    # Green half becomes a pull-quote panel; red quarter becomes a bullet.
    p = layout_p(t, 2)
    q = tuple(lerp(ASYM[3][i], QUOTE[i], p) for i in range(4))
    pq = clear_p(t, 0.2, 0.6)
    q = (q[0], lerp(q[1], q[3], pq), q[2], q[3])
    if q[3] > q[1]:
        fill_box(ctx, *q, GREEN)
        ctx.save()
        clip_box(ctx, *q)
        for i, bb in enumerate(QUOTE_BARS):
            draw_bar(ctx, bb, prog(t, T_TEXT + 0.6 + 0.08 * i, 0.6, in_out_cubic), PAPER)
        ctx.restore()
    p = layout_p(t, 3)
    r = tuple(lerp(ORDER_CELLS[4][i], BULLET[i], p) for i in range(4))
    k = 1 - clear_p(t, 0.0, 0.4)
    if k > 0:
        cxr, cyr = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        fill_box(ctx, cxr - (r[2] - r[0]) * k / 2, cyr - (r[3] - r[1]) * k / 2,
                 cxr + (r[2] - r[0]) * k / 2, cyr + (r[3] - r[1]) * k / 2, RED)

    k = 1 - clear_p(t, 0.05, 0.5)
    for i, bb in enumerate(CAPTION_BARS):
        draw_bar(ctx, bb, prog(t, T_TEXT + 0.3 + 0.08 * i, 0.6, in_out_cubic) * k, INK)
    rule = (X[4] + G, Y[1] + 14, X[8], Y[1] + 18)
    draw_bar(ctx, rule, prog(t, T_CLARITY + 0.4, 0.7, in_out_cubic) * k, INK)


def clarity_line():
    tl = ty.line(CL_FONT, "CLARITY", CL_SIZE)
    return tl, tl.x_for(CL_X, "left"), CL_BASE


def draw_clarity(ctx, t):
    if t < T_CLARITY:
        return
    tl, ox, base = clarity_line()
    n = len(tl.glyphs)
    if t < T_CL_FILL + 0.06 * n + 0.4:
        ctx.save()
        clip_box(ctx, X[4], 0, W, base + 6)

        def rise(i):
            return (1 - prog(t, T_CLARITY + i * 0.05, 0.7, out_quart)) * (tl.cap + 20)

        draw_filling_word(ctx, t, tl, ox, base, T_CL_FILL, BAND_COLORS, rise, step=0.06)
        ctx.restore()
    for k in range(n):
        if fill_progress(t, T_CL_FILL, k, 0.06) < 1:
            continue
        start = glyph_box(tl, k, ox, base)
        col = (X[0] + MOD * k, Y[0], X[1] + MOD * k, Y[4])
        ts = T_CL_SNAP + k * BEAT / 4
        tb = T_BANDS + k * 0.04
        box = track(t, [(ts, start), (ts + 0.6, col, out_expo), (tb, col), (tb + 1.3, BANDS[k], in_out_quart)])
        # One pixel of overlap hides seams between neighbouring bands.
        fill_box(ctx, box[0], box[1], box[2] + (1 if t >= tb else 0), box[3], BAND_COLORS[k])


def draw(ctx, t):
    a = grid_alpha(t)
    s = prog(t, T_LAYOUT, 0.6, in_out_cubic)
    draw_grid(ctx, t, a * s)          # beneath the layout
    draw_layout(ctx, t)
    draw_order(ctx, t)
    draw_grid(ctx, t, a * (1 - s))    # on top of the blocks before the layout
    draw_balance(ctx, t)
    draw_clarity(ctx, t)
