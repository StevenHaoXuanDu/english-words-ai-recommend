"""Scene 08 - The Final Composition (02:45-03:00).

The film's vocabulary returns and comes to rest on a single rule: the
blue square, the black circle, the yellow square, the red dot and - rising
from behind the line exactly as the opening circle sank into it - a green
semicircle. Below, LESS, BUT BETTER. and its caption. The composition is
complete and still from T_HOLD; sgd.film fades it to black at 178.0 s.
"""

from .. import typography as ty
from ..core import *  # noqa: F401,F403

START, END = 165.0, 180.0

T_RULE = bt(66)          # 165.0   the rule
T_ELEMS = bt(66, 1)      # 165.625 elements return, one per beat
T_TEXT = bt(68)          # 170.0   LESS, BUT BETTER.
T_CAPTION = bt(68, 3)    # 171.875 CLARITY · FUNCTION · FORM
T_HOLD = bt(69, 2)       # 173.75  everything still

RULE_Y = Y[2]
BLUE_SQ = (X[0], Y[0], X[2], Y[2])
CIRCLE = (X[3], Y[1], MOD)
YELLOW_SQ = (X[4], Y[1], X[5], Y[2])
RED_DOT = (X[5] + HALF, Y[2] - HALF / 2, HALF / 2)
DOME = (X[7], Y[2], MOD)

TEXT = "LESS, BUT BETTER."
TEXT_SIZE = ty.fit_size(ty.DISPLAY_BOLD, TEXT, X[6] - X[0], -0.01)
TEXT_BASE = Y[3]
CAPTION = "CLARITY · FUNCTION · FORM"
CAPTION_BASE = Y[3] + HALF

EVENTS = [
    (T_RULE, "glide", {"dur": 1.0}),
    (T_ELEMS + 0.55, "thud", {}),
    (T_ELEMS + BEAT + 0.55, "bloom", {"note": 45}),
    (T_ELEMS + 2 * BEAT + 0.4, "click", {"pitch": 1.2}),
    (T_ELEMS + 3 * BEAT, "bell", {"note": 86}),
    (T_ELEMS + 4 * BEAT, "sub", {"note": 41}),
    (T_TEXT, "chime", {"note": 77}),
    (T_TEXT + BEAT / 2, "chime", {"note": 81}),
    (T_TEXT + BEAT, "chime", {"note": 84}),
    (T_CAPTION, "tick", {}),
]


def elem_p(t, k, dur=1.25, ease=out_quart):
    return prog(t, T_ELEMS + k * BEAT, dur, ease)


def draw_text(ctx, t):
    if t < T_TEXT:
        return
    tl = ty.line(ty.DISPLAY_BOLD, TEXT, TEXT_SIZE, -0.01)
    ox = tl.x_for(X[0], "left")
    word = 0
    rise = tl.cap + 60
    ctx.save()
    clip_box(ctx, 0, TEXT_BASE - tl.cap - 30, W, TEXT_BASE + 0.2 * TEXT_SIZE)
    for i, g in enumerate(tl.glyphs):
        if g.char == " ":
            word += 1
            continue
        dy = (1 - prog(t, T_TEXT + word * BEAT / 2, 0.85, out_quart)) * rise
        tl.glyph_path(ctx, i, ox, TEXT_BASE + dy)
    set_color(ctx, INK)
    ctx.fill()
    ctx.restore()

    if t < T_CAPTION:
        return
    cl = ty.line(ty.TEXT_MEDIUM, CAPTION, 26, 0.24)
    dy = (1 - prog(t, T_CAPTION, 0.9, out_quart)) * (cl.cap + 16)
    ctx.save()
    clip_box(ctx, 0, CAPTION_BASE - cl.cap - 8, W, CAPTION_BASE + 6)
    cl.draw(ctx, cl.x_for(X[0], "left"), CAPTION_BASE + dy, INK)
    ctx.restore()


def draw(ctx, t):
    # The rule draws outwards from the centre.
    p = prog(t, T_RULE, 1.0, in_out_cubic)
    half = (X[8] - X[0]) / 2 * p
    if half > 0:
        hline(ctx, CX - half, CX + half, RULE_Y, 2, INK)

    # Blue square slides in from the left.
    p = elem_p(t, 0)
    if p > 0:
        dx = (1 - p) * -(BLUE_SQ[2] + 40)
        fill_box(ctx, BLUE_SQ[0] + dx, BLUE_SQ[1], BLUE_SQ[2] + dx, BLUE_SQ[3], BLUE)

    # Black circle descends onto the rule.
    p = elem_p(t, 1)
    if p > 0:
        cx, cy, r = CIRCLE
        fill_circle(ctx, cx, lerp(-r - 30, cy, p), r, INK)

    ctx.save()
    clip_box(ctx, 0, 0, W, RULE_Y - 1)
    # Yellow square rises from behind the rule.
    p = elem_p(t, 2)
    if p > 0:
        dy = (1 - p) * (YELLOW_SQ[3] - YELLOW_SQ[1])
        fill_box(ctx, YELLOW_SQ[0], YELLOW_SQ[1] + dy, YELLOW_SQ[2], YELLOW_SQ[3] + dy, YELLOW)
    # Green semicircle rises from behind the rule - the opening, reversed.
    p = elem_p(t, 4, 1.6, in_out_cubic)
    if p > 0:
        cx, cy, r = DOME
        fill_circle(ctx, cx, cy + (1 - p) * r, r, GREEN)
    ctx.restore()

    # Red dot.
    p = elem_p(t, 3, 0.6, out_cubic)
    if p > 0:
        cx, cy, r = RED_DOT
        fill_circle(ctx, cx, cy, r * p, RED)

    draw_text(ctx, t)
