"""Scene 06 - Systems and Patterns (02:00-02:25).

The blue square from Scene 05 receives a motif - a yellow quarter disc -
and duplicates ring by ring into a full-bleed pattern. Its tiles turn,
then the pattern becomes a visual identity applied across formats:

    poster -> packaging dieline -> magazine spread -> digital interface

Every format is described by the same roles (frames, pattern area, title,
text bars, accent), so each one morphs into the next, keeping the same
grid, palette and geometric vocabulary. All four end up side by side.
"""

import math

from ..core import *  # noqa: F401,F403

START, END = 120.0, 145.0

T_MOTIF = bt(48, 1)     # 120.625 the square receives its motif
T_DUP = bt(48, 3)       # 121.875 it duplicates outwards, ring by ring
T_TURN = bt(49, 3)      # 124.375 tiles turn a quarter
T_POSTER = bt(50, 2)    # 126.25  pattern -> poster
T_PACK = bt(52)         # 130.0   poster -> packaging
T_SPREAD = bt(53, 2)    # 133.75  packaging -> magazine spread
T_UI = bt(55)           # 137.5   spread -> digital interface
T_SYSTEM = bt(56, 2)    # 141.25  all formats side by side
T_OUT = bt(57, 2)       # 143.75  clear for Scene 07

FULL = (-195.0, -195.0, 210.0)          # pattern origin x, y and tile size
MOTIF = 0.88                             # quarter-disc radius, fraction of the tile
CENTER_TILE = (5, 3)

EVENTS = (
    [(T_MOTIF, "bell", {"note": 74})]
    + [(T_DUP + r * BEAT / 2, "snap", {"pitch": 0.8 + 0.12 * r}) for r in range(5)]
    + [(T_TURN, "ticks", {"n": 12, "step": 0.05}),
       (T_POSTER, "low", {"note": 38}),
       (T_POSTER + 1.0, "ticks", {"n": 4, "step": 0.08}),
       (T_PACK, "low", {"note": 41}),
       (T_PACK + 0.6, "ticks", {"n": 6, "step": 0.06}),
       (T_SPREAD, "low", {"note": 36}),
       (T_UI, "low", {"note": 43}),
       (T_UI + 0.9, "click", {"pitch": 1.5}),
       (T_SYSTEM, "chime", {"note": 79}),
       (T_SYSTEM + 0.3, "chime", {"note": 84}),
       (T_OUT, "swell", {"dur": 1.2})]
)


# --------------------------------------------------------------------------
# pattern
# --------------------------------------------------------------------------

def rule_angle(i, j):
    """Quarter-disc orientation: four tiles around every other crossing
    point their discs at it, forming full circles."""
    return {(0, 0): 2, (1, 0): 3, (0, 1): 1, (1, 1): 0}[(i % 2, j % 2)] * math.pi / 2


def tile_angle(t, i, j):
    """Each tile turns half a revolution in a diagonal wave; the circles
    re-form around the alternate crossings."""
    turn = prog(t, T_TURN + (i + j) * 0.035, 0.75, in_out_cubic)
    return rule_angle(i, j) + turn * math.pi


def draw_tile(ctx, x, y, s, angle, motif=1.0):
    fill_rect(ctx, x, y, s, s, BLUE)
    if motif <= 0:
        return
    ctx.save()
    ctx.rectangle(x, y, s, s)
    ctx.clip()
    ctx.translate(x + s / 2, y + s / 2)
    ctx.rotate(angle)
    fill_sector(ctx, -s / 2, -s / 2, s * MOTIF * motif, 0, math.pi / 2, YELLOW)
    ctx.restore()


def draw_pattern(ctx, t, box, origin):
    """Fill box with the settled pattern whose tile (0, 0) sits at origin
    (x, y, size). Once the tiles have turned, every four quarter discs form
    a full circle on the crossings with even indices, so whole circles are
    drawn - no seams between tiles at any scale."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 0.5 or y1 - y0 < 0.5:
        return
    ox, oy, s = origin
    r = s * MOTIF
    ctx.save()
    clip_box(ctx, x0, y0, x1, y1)
    fill_box(ctx, x0, y0, x1, y1, BLUE)
    i0, i1 = int(math.floor((x0 - r - ox) / s)), int(math.ceil((x1 + r - ox) / s))
    j0, j1 = int(math.floor((y0 - r - oy) / s)), int(math.ceil((y1 + r - oy) / s))
    for j in range(j0 - j0 % 2, j1 + 1, 2):
        for i in range(i0 - i0 % 2, i1 + 1, 2):
            circle_path(ctx, ox + i * s, oy + j * s, r)
    set_color(ctx, YELLOW)
    ctx.fill()
    ctx.restore()


def draw_duplication(ctx, t):
    """Single square -> motif -> rings of copies sliding outwards."""
    ci, cj = CENTER_TILE
    ox, oy, s = FULL
    for r in range(5, -1, -1):
        tr = T_DUP + (r - 1) * BEAT / 2
        if r > 0 and t < tr:
            continue
        p = 1.0 if r == 0 else prog(t, tr, 0.45, out_cubic)
        for j in range(7):
            for i in range(11):
                if max(abs(i - ci), abs(j - cj)) != r:
                    continue
                di = (1 if i > ci else -1) if abs(i - ci) == r else 0
                dj = (1 if j > cj else -1) if abs(j - cj) == r else 0
                x = ox + (i - di * (1 - p)) * s
                y = oy + (j - dj * (1 - p)) * s
                motif = prog(t, T_MOTIF, 0.8, out_cubic) if r == 0 else 1.0
                draw_tile(ctx, x, y, s, tile_angle(t, i, j), motif)


# --------------------------------------------------------------------------
# formats, described by shared roles
# --------------------------------------------------------------------------

def bars(x, y, widths, pitch, h):
    return [(x, y + k * pitch, x + w, y + k * pitch + h) for k, w in enumerate(widths)]


POSTER = {
    "frames": [(750, 225, 1170, 855)],
    "pattern": ((750, 225, 1170, 645), (540, 155, 70)),
    "title": (780, 677, 1060, 709),
    "bars": bars(780, 729, (220, 250, 160), 18, 8),
    "accent": (1128, 812, 15, 15, 1.0),       # cx, cy, half w, half h, roundness
}

# Box dieline: glue | side | front | side | back, flaps on front and back.
_GL, _SL, _FR, _SR, _BK = (520, 560), (560, 680), (680, 980), (980, 1100), (1100, 1400)
_TOP, _BODY, _BOT = (195, 315), (315, 765), (765, 885)
PACK = {
    "frames": [(_FR[0], _BODY[0], _FR[1], _BODY[1]),
               (_SL[0], _BODY[0], _SL[1], _BODY[1]), (_SR[0], _BODY[0], _SR[1], _BODY[1]),
               (_BK[0], _BODY[0], _BK[1], _BODY[1]), (_GL[0], _BODY[0] + 30, _GL[1], _BODY[1] - 30),
               (_FR[0], _TOP[0], _FR[1], _TOP[1]), (_FR[0], _BOT[0], _FR[1], _BOT[1]),
               (_BK[0], _TOP[0], _BK[1], _TOP[1]), (_BK[0], _BOT[0], _BK[1], _BOT[1]),
               (_SL[0] + 10, _TOP[1] - 70, _SL[1] - 10, _TOP[1]), (_SR[0] + 10, _TOP[1] - 70, _SR[1] - 10, _TOP[1]),
               (_SL[0] + 10, _BOT[0], _SL[1] - 10, _BOT[0] + 70), (_SR[0] + 10, _BOT[0], _SR[1] - 10, _BOT[0] + 70)],
    "pattern": ((_SL[0], _BODY[0], _BK[1], _BODY[0] + 240), (560, 255, 60)),
    "title": (705, 582, 930, 606),
    "bars": bars(705, 624, (175, 205, 125), 16, 7),
    "accent": (940, 725, 18, 18, 1.0),
    "folds": True,
}

SPREAD = {
    "frames": [(960, 210, 1620, 870), (300, 210, 960, 870)],
    "pattern": ((300, 210, 960, 870), (300, 144, 66)),
    "title": (1010, 270, 1400, 310),
    "bars": (bars(1010, 330, (300,), 0, 12)
             + bars(1010, 392, (270, 270, 270, 240, 270, 270, 150, 0, 270, 270, 270, 210, 270, 270, 270, 120), 24, 7)
             + bars(1320, 392, (250, 250, 220, 250, 250, 250, 250, 90, 0, 250, 250, 250, 250, 180), 24, 7)),
    "accent": (1555, 290, 34, 34, 1.0),
}

UI = {
    "frames": [(360, 150, 1560, 930)],
    "pattern": ((400, 280, 1520, 560), (400, 210, 70)),
    "title": (400, 604, 870, 642),
    "bars": bars(400, 662, (380, 420, 300), 20, 8),
    "accent": (490, 765, 90, 25, 1.0),
    "chrome": True,
}

FORMATS = (POSTER, PACK, SPREAD, UI)


def _lerp_box(a, b, p):
    return tuple(a[i] + (b[i] - a[i]) * p for i in range(len(a)))


def _lerp_list(a, b, p, seed_a, seed_b):
    """Interpolate two lists of boxes; extras grow from / shrink to a seed."""
    n = max(len(a), len(b))
    out = []
    for k in range(n):
        ba = a[k] if k < len(a) else seed_a
        bb = b[k] if k < len(b) else seed_b
        out.append(_lerp_box(ba, bb, p))
    return out


def _collapse(box, mode="left"):
    x0, y0, x1, y1 = box
    if mode == "left":
        return (x0, y0, x0, y1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx, cy, cx, cy)


def _hinge(box, anchor):
    """A panel folded flat against the side of `anchor` it is attached to,
    so it can unfold from that edge (or fold back onto it)."""
    x0, y0, x1, y1 = box
    ax0, ay0, ax1, ay1 = anchor
    if x1 <= ax0 + 1:
        return (x1, y0, x1, y1)
    if x0 >= ax1 - 1:
        return (x0, y0, x0, y1)
    if y1 <= ay0 + 1:
        return (x0, y1, x1, y1)
    if y0 >= ay1 - 1:
        return (x0, y0, x1, y0)
    return _collapse(box, "center")


def _lerp_frames(fa, fb, p):
    """Frames that only exist on one side unfold from / fold onto the edge
    of the main panel, like the flaps of a box."""
    out = [_lerp_box(fa[0], fb[0], p)]
    for k in range(1, max(len(fa), len(fb))):
        if k < len(fa) and k < len(fb):
            out.append(_lerp_box(fa[k], fb[k], p))
        elif k < len(fb):
            out.append(_lerp_box(_hinge(fb[k], fb[0]), fb[k], p))
        else:
            out.append(_lerp_box(fa[k], _hinge(fa[k], fa[0]), p))
    return out


def mix_formats(a, b, p):
    """Staged morph: text and accent retract, frames and pattern move, then
    the text grows back in its new place. Returns (state, build)."""
    pf = in_out_cubic(clamp01((p - 0.1) / 0.8))
    pm = clamp01((p - 0.35) / 0.3)
    build = 1 - in_out_cubic(clamp01(p / 0.35)) if p < 0.5 else in_out_cubic(clamp01((p - 0.65) / 0.35))
    fa, fb = a["frames"], b["frames"]
    out = {
        "frames": _lerp_frames(fa, fb, pf),
        "pattern": (_lerp_box(a["pattern"][0], b["pattern"][0], pf), _lerp_box(a["pattern"][1], b["pattern"][1], pf)),
        "title": _lerp_box(a["title"], b["title"], pm),
        "bars": _lerp_list(a["bars"], b["bars"], pm, _collapse(a["title"]), _collapse(b["title"])),
        "accent": _lerp_box(a["accent"], b["accent"], pm),
        "folds": (clamp01((p - 0.6) / 0.4) if b.get("folds") else 0.0)
                 + ((1 - clamp01(p / 0.4)) if a.get("folds") else 0.0),
        "chrome": (clamp01((p - 0.6) / 0.4) if b.get("chrome") else 0.0)
                  + ((1 - clamp01(p / 0.4)) if a.get("chrome") else 0.0),
    }
    return out, build


def draw_format(ctx, t, st, build=1.0):
    frames = st["frames"]
    # Paper panels and their outline.
    for b in frames:
        if b[2] - b[0] > 0.5 and b[3] - b[1] > 0.5:
            fill_box(ctx, *b, PAPER)
    pbox, origin = st["pattern"]
    draw_pattern(ctx, t, pbox, origin)
    chrome = st.get("chrome", 0.0)
    for b in frames:
        w, h = b[2] - b[0], b[3] - b[1]
        if w > 0.5 and h > 0.5:
            ctx.rectangle(b[0], b[1], w, h)
            ctx.set_line_width(2)
            set_color(ctx, INK)
            ctx.stroke()
    folds = st.get("folds", 0.0)
    if folds > 0.01:
        a = folds
        for x in (_SL[0], _SL[1], _FR[1], _SR[1]):
            dashed_line(ctx, x, _BODY[0], x, _BODY[1], 2, INK, 9, 7, a)
        for y in (_BODY[0], _BODY[1]):
            dashed_line(ctx, _FR[0], y, _FR[1], y, 2, INK, 9, 7, a)
            dashed_line(ctx, _BK[0], y, _BK[1], y, 2, INK, 9, 7, a)
    if chrome > 0.01:
        f = frames[0]
        hline(ctx, f[0], f[2], f[1] + 50, 2, INK, chrome)
        for k in range(3):
            fill_circle(ctx, f[0] + 30 + 22 * k, f[1] + 25, 6.5, INK, chrome)
        fill_rrect(ctx, f[0] + 120, f[1] + 13, 480, 24, 12, INK, 0.10 * chrome)
        fill_rect(ctx, 400, f[1] + 78, 26, 26, BLUE, chrome)
        for k, w in enumerate((70, 90, 60)):
            fill_rect(ctx, 1240 + k * 100, f[1] + 88, w, 7, INK, chrome)
        # Three cards carrying the pattern as thumbnails.
        for k in range(3):
            x = 940 + k * 200
            card = (x, 604, x + 180, 724)
            if chrome > 0:
                ctx.push_group()
                draw_pattern(ctx, t, card, (x, 604 - 30, 30))
                ctx.pop_group_to_source()
                ctx.paint_with_alpha(chrome)
            ctx.rectangle(x, 604, 180, 120)
            ctx.set_line_width(2)
            set_color(ctx, INK, chrome)
            ctx.stroke()
            fill_rect(ctx, x, 744, 150, 7, INK, chrome)
            fill_rect(ctx, x, 762, 110, 7, INK, chrome)
    tb = st["title"]
    fill_box(ctx, tb[0], tb[1], tb[0] + (tb[2] - tb[0]) * build, tb[3], INK)
    for k, b in enumerate(st["bars"]):
        if b[2] - b[0] > 0.5:
            fill_box(ctx, b[0], b[1], b[0] + (b[2] - b[0]) * build, b[3], INK)
    cx, cy, hw, hh, rnd = st["accent"]
    fill_morph(ctx, cx, cy, 2 * hw * build, 2 * hh * build, rnd, RED)
    if chrome > 0.01:
        fill_rect(ctx, cx - hw + 30, cy - 3.5, 2 * hw - 60, 7, PAPER, chrome)


def format_state(t):
    """Current (possibly mid-transition) format and its text build."""
    st, build = POSTER, 1.0
    for k, tk in enumerate((T_PACK, T_SPREAD, T_UI)):
        p = prog(t, tk, 1.5, linear)
        if p <= 0:
            break
        st, build = mix_formats(FORMATS[k], FORMATS[k + 1], p) if p < 1 else (FORMATS[k + 1], 1.0)
    return st, build


def poster_build(t):
    """Pattern contracts from full bleed into the poster's pattern area."""
    p = prog(t, T_POSTER, 1.25, in_out_quart)
    pbox = _lerp_box((0, 0, W, H), POSTER["pattern"][0], p)
    origin = _lerp_box(FULL, POSTER["pattern"][1], p)
    st = dict(POSTER)
    st["pattern"] = (pbox, origin)
    pf = prog(t, T_POSTER + 0.6, 0.7, in_out_cubic)
    fb = POSTER["frames"][0]
    st["frames"] = [(fb[0], fb[1], fb[2], lerp(POSTER["pattern"][0][3], fb[3], pf))] if pf > 0 else []
    return st, prog(t, T_POSTER + 1.0, 0.6, in_out_cubic)


# --------------------------------------------------------------------------
# overview: all four formats side by side
# --------------------------------------------------------------------------
SLOTS = (  # (format, centre x, centre y, scale)
    (POSTER, 246, 540, 0.60),
    (PACK, 610, 540, 0.45),
    (SPREAD, 1086, 540, 0.36),
    (UI, 1584, 540, 0.36),
)


def format_center(f):
    xs = [b[0] for b in f["frames"]] + [b[2] for b in f["frames"]]
    ys = [b[1] for b in f["frames"]] + [b[3] for b in f["frames"]]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def draw_overview(ctx, t):
    out = prog(t, T_OUT, 1.0, in_out_quart)
    for k, (f, sx, sy, sc) in enumerate(SLOTS):
        fc = format_center(f)
        if f is UI:
            p = prog(t, T_SYSTEM, 1.2, in_out_quart)
            scale = lerp(1.0, sc, p)
            cx, cy = lerp(fc[0], sx, p), lerp(fc[1], sy, p)
        else:
            p = prog(t, T_SYSTEM + 0.25 + 0.15 * k, 0.9, out_cubic)
            if p <= 0:
                continue
            scale = sc * p
            cx, cy = sx, sy
        scale *= 1 - out
        if scale <= 0.002:
            continue
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(scale, scale)
        ctx.translate(-fc[0], -fc[1])
        draw_format(ctx, t, f)
        ctx.restore()


def draw(ctx, t):
    if t < T_POSTER:
        draw_duplication(ctx, t)
        return
    if t < T_PACK:
        st, build = poster_build(t)
        draw_format(ctx, t, st, build)
        return
    if t < T_SYSTEM:
        st, build = format_state(t)
        draw_format(ctx, t, st, build)
        return
    draw_overview(ctx, t)
