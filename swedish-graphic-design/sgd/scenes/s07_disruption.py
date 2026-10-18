"""Scene 07 - Controlled Disruption (02:25-02:45).

A calm, point-symmetric composition on the 8 x 4 grid is broken: the
circle outgrows its cell, rectangles slip out of alignment, the grid
lines drift and a heavy black line cuts across the frame. The cut runs at
a slope of 1 : 2 through the centre, so it passes exactly through grid
crossings - and in the reorganisation the whole grid turns onto it. The
disruption becomes the new axis: every element snaps to the rotated grid.

All positions are in grid coordinates (u, v) relative to the frame centre;
the grid itself rotates by phi(t).
"""

import math

from ..core import *  # noqa: F401,F403

START, END = 145.0, 165.0

T_GRID = bt(58)           # 145.0   grid draws
T_BUILD = bt(58, 1)       # 145.625 elements arrive
T_CIRCLE = bt(60)         # 150.0   circle moves beyond its cell
T_SLIP = bt(60, 1)        # 150.625 rectangles slip out of alignment
T_DRIFT = bt(60, 2)       # 151.25  grid lines drift
T_CUT = bt(60, 3)         # 151.875 a black line cuts through
T_REORG = bt(62)          # 155.0   grid turns onto the cut, elements snap
T_EXIT = bt(65)           # 162.5   elements leave along the new axis

THETA = math.atan2(-1, 2)              # the cut: slope -1/2 through the centre
CUT_W = 10.0

# (kind, colour, (u0, v0, u1, v1) or (cu, cv, r)) in the calm composition
CALM = {
    "blue": (-840, -420, -420, 0),
    "yellow": (420, 0, 840, 420),
    "red": (630, -420, 840, -210),
    "green": (-840, 210, -630, 420),
    "circle": (0, 0, 210),
}
# Disrupted offsets: (du, dv, rotation)
SLIP = {
    "blue": (60, 38, math.radians(-8)),
    "yellow": (-52, -34, math.radians(6)),
    "red": (44, 86, math.radians(15)),
    "green": (74, -54, math.radians(-12)),
}
CIRCLE_OUT = (300, -60, 290)
# The new arrangement on the rotated grid.
NEW = {
    "blue": (-630, -420, -210, 0),
    "yellow": (0, 0, 420, 420),
    "red": (525, -105, 630, 0),
    "green": (-630, 0, -420, 210),
    "circle": (210, -210, 210),
}
EXIT_DIR = {"blue": -1, "green": -1, "yellow": 1, "red": 1, "circle": 1}
COLORS = {"blue": BLUE, "yellow": YELLOW, "red": RED, "green": GREEN}
DRIFT_X = (0, 18, -26, 12, 0, -20, 30, -14, 0)
DRIFT_Y = (0, -16, 0, 22, 0)
TILT_X = (0, 1.2, -0.8, 0.5, 0, -1.5, 0.9, -0.6, 0)

EVENTS = [
    (T_GRID, "ticks", {"n": 13, "step": 0.05}),
    (T_BUILD, "tick", {}),
    (T_BUILD + BEAT / 2, "bloom", {"note": 45}),
    (T_BUILD + BEAT, "thud", {}), (T_BUILD + 1.5 * BEAT, "thud", {}),
    (T_BUILD + 2 * BEAT, "click", {"pitch": 1.3}), (T_BUILD + 2.5 * BEAT, "click", {"pitch": 1.1}),
    (T_CIRCLE, "tension", {"dur": 5.0}),
    (T_SLIP, "thud", {}),
    (T_DRIFT, "glide", {"dur": 0.8}),
    (T_CUT, "hit", {}),
    (T_REORG, "low", {"note": 41}),
    (T_REORG + 0.3125, "snap", {"pitch": 0.9}), (T_REORG + 0.625, "snap", {"pitch": 1.1}),
    (T_REORG + 0.9375, "snap", {"pitch": 1.3}), (T_REORG + 1.25, "snap", {"pitch": 1.5}),
    (T_REORG + 1.5625, "chime", {"note": 81}),
    (T_EXIT, "swell", {"dur": 2.0}),
]


def grid_phi(t):
    return THETA * prog(t, T_REORG, 1.5, in_out_quart)


def disorder(t):
    """0 -> 1 as the composition breaks, back to 0 as it reorganises."""
    return prog(t, T_DRIFT, 0.8, in_out_cubic) * (1 - prog(t, T_REORG, 1.2, in_out_quart))


def to_world(u, v, phi):
    c, s = math.cos(phi), math.sin(phi)
    return CX + u * c - v * s, CY + u * s + v * c


def exit_shift(t, name):
    p = prog(t, T_EXIT + (0.0 if EXIT_DIR[name] < 0 else 0.25), 1.6, in_cubic)
    return EXIT_DIR[name] * 1700 * p


def rect_state(t, name, k):
    """Centre (u, v), half sizes and rotation of a rectangle in grid space."""
    u0, v0, u1, v1 = CALM[name]
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    hw, hh = (u1 - u0) / 2, (v1 - v0) / 2
    tb = T_BUILD + BEAT * (1, 1.5, 2, 2.5)[k]
    p_in = prog(t, tb, 0.8, out_quart)
    if name in ("blue", "yellow"):
        cu += (-1 if name == "blue" else 1) * 1100 * (1 - p_in)
        scale = 1.0
    else:
        scale = p_in
    du, dv, rot = SLIP[name]
    ps = prog(t, T_SLIP + 0.1 * k, 0.9, in_out_cubic)
    creep = prog(t, T_CUT, 3.0, linear) * 0.04 * (1 if k % 2 else -1)
    cu, cv, rot = cu + du * ps, cv + dv * ps, rot * ps + creep * ps
    n0, nv0, n1, nv1 = NEW[name]
    pr = prog(t, T_REORG + 0.3125 * (k + 1), 1.25, in_out_quart)
    cu = lerp(cu, (n0 + n1) / 2, pr)
    cv = lerp(cv, (nv0 + nv1) / 2, pr)
    hw, hh = lerp(hw, (n1 - n0) / 2, pr), lerp(hh, (nv1 - nv0) / 2, pr)
    rot = lerp(rot, 0.0, pr)
    cu += exit_shift(t, name)
    return cu, cv, hw * scale, hh * scale, rot


def circle_state(t):
    cu, cv, r = CALM["circle"]
    r *= prog(t, T_BUILD + BEAT / 2, 0.7, out_cubic)
    po = prog(t, T_CIRCLE, 1.3, in_out_cubic)
    cu, cv, r = lerp(cu, CIRCLE_OUT[0], po), lerp(cv, CIRCLE_OUT[1], po), lerp(r, CIRCLE_OUT[2], po)
    pr = prog(t, T_REORG + 0.3125 * 2.5, 1.25, in_out_quart)
    n = NEW["circle"]
    cu, cv, r = lerp(cu, n[0], pr), lerp(cv, n[1], pr), lerp(r, n[2], pr)
    return cu + exit_shift(t, "circle"), cv, r


def rect_path(ctx, cu, cv, hw, hh, rot, phi):
    ctx.save()
    x, y = to_world(cu, cv, phi)
    ctx.translate(x, y)
    ctx.rotate(phi + rot)
    ctx.rectangle(-hw, -hh, 2 * hw, 2 * hh)
    ctx.restore()


def draw_grid(ctx, t, phi):
    fade = 1 - prog(t, T_EXIT, 1.2, in_out_cubic)
    a = 0.30 * fade
    if a <= 0:
        return
    dis = disorder(t)
    ctx.save()
    ctx.translate(CX, CY)
    ctx.rotate(phi)
    for i, x in enumerate(X):
        p = prog(t, T_GRID + 0.04 * i, 0.8, in_out_quart)
        if p <= 0:
            continue
        ctx.save()
        ctx.translate(x - CX + DRIFT_X[i] * dis, 0)
        ctx.rotate(math.radians(TILT_X[i]) * dis)
        fill_rect(ctx, -1, -420, 2, 840 * p, INK, a)
        ctx.restore()
    for j, y in enumerate(Y):
        p = prog(t, T_GRID + 0.3 + 0.04 * j, 0.8, in_out_quart)
        if p <= 0:
            continue
        fill_rect(ctx, -840, y - CY - 1 + DRIFT_Y[j] * dis, 1680 * p, 2, INK, a)
    ctx.restore()


def draw_cut(ctx, t, phi):
    """The heavy diagonal. Drawn fast, it stays as the new axis."""
    p = prog(t, T_CUT, 0.32, out_expo)
    if p <= 0:
        return
    retract = prog(t, T_EXIT + 0.5, 1.4, in_out_cubic)
    width = lerp(CUT_W, 8.0, prog(t, T_REORG, 1.5, in_out_cubic))
    length = 1250.0
    a0 = -length + 2 * length * retract * 0.5
    a1 = -length + 2 * length * p - 2 * length * retract * 0.5
    if a1 <= a0:
        return
    x0, y0 = to_world(a0, 0, THETA)
    x1, y1 = to_world(a1, 0, THETA)
    line(ctx, x0, y0, x1, y1, width, INK)


def draw_axis(ctx, t, phi):
    """The calm composition's horizontal axis; it turns with the grid."""
    p = prog(t, T_BUILD, 0.9, in_out_quart) * (1 - prog(t, T_EXIT, 1.0, in_out_cubic))
    if p <= 0:
        return
    tilt = math.radians(-3) * disorder(t)
    ctx.save()
    ctx.translate(CX, CY)
    ctx.rotate(phi + tilt)
    fill_rect(ctx, -840 * p, -1, 1680 * p, 2, INK)
    ctx.restore()


def draw(ctx, t):
    phi = grid_phi(t)
    draw_grid(ctx, t, phi)
    draw_axis(ctx, t, phi)

    rects = []
    for k, name in enumerate(("blue", "yellow", "red", "green")):
        st = rect_state(t, name, k)
        if st[2] <= 0 or st[3] <= 0:
            continue
        rect_path(ctx, *st, phi)
        set_color(ctx, COLORS[name])
        ctx.fill()
        rects.append(st)

    cu, cv, r = circle_state(t)
    if r > 0:
        x, y = to_world(cu, cv, phi)
        fill_circle(ctx, x, y, r, INK)
        # Where the circle overlaps a rectangle it turns into negative space.
        for st in rects:
            ctx.save()
            rect_path(ctx, *st, phi)
            ctx.clip()
            fill_circle(ctx, x, y, r, PAPER)
            ctx.restore()

    draw_cut(ctx, t, phi)
