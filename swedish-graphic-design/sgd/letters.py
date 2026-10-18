"""A parametric, constructed geometric alphabet.

Each capital is assembled from two kinds of pieces:

    ("q", x0, y0, x1, y1, x2, y2, x3, y3)   a quad (stems, arms, diagonals),
                                           vertices clockwise from top left
    ("a", cx, cy, r_out, r_in, a0, a1)      a ring segment (bowls, rounds)

Because letters are plain geometry they can be taken apart into abstract
forms, morphed into other letters and given any weight (stroke) or scale.
"""

import math

from .core import lerp, ring_seg_path

TAU = 2 * math.pi
OVER = 0.016   # optical overshoot of round forms, fraction of cap height


def R(x, y, w, h):
    return ("q", x, y, x + w, y, x + w, y + h, x, y + h)


def Q(*pts):
    out = ["q"]
    for p in pts:
        out.extend(p)
    return tuple(out)


def A(cx, cy, ro, ri, a0, a1):
    return ("a", cx, cy, ro, max(ri, 0.0), a0, a1)


def _bowl(h, s, w, b):
    """Stem, two arms and a half ring: the shared top of P and R."""
    a = w - b
    ri = max(b - s, 0.02 * h)
    return [R(0, 0, s, h), R(s * 0.5, 0, a - s * 0.5, s), R(s * 0.5, 2 * b - s, a - s * 0.5, s),
            A(a, b, b, ri, -math.pi / 2, math.pi / 2)]


def glyph(ch, h, s):
    """Pieces for one capital with cap height h and stroke s.

    Returns (advance width, left bearing, right bearing, pieces) with the
    origin at the top left of the cap box, y pointing down.
    """
    if ch == "F":
        w = 0.58 * h
        return w, .07, .02, [R(0, 0, s, h), R(s * .5, 0, w - s * .5, s),
                             R(s * .5, 0.47 * h - s / 2, 0.84 * w - s * .5, s)]
    if ch == "E":
        w = 0.58 * h
        return w, .07, .03, [R(0, 0, s, h), R(s * .5, 0, w - s * .5, s),
                             R(s * .5, 0.5 * h - s / 2, 0.88 * w - s * .5, s),
                             R(s * .5, h - s, w - s * .5, s)]
    if ch == "T":
        w = 0.70 * h
        return w, .02, .02, [R(0, 0, w, s), R(w / 2 - s / 2, s * .5, s, h - s * .5)]
    if ch == "I":
        return s, .07, .07, [R(0, 0, s, h)]
    if ch == "O":
        r = h / 2 + OVER * h
        return h, .03, .03, [A(h / 2, h / 2, r, r - s, -math.pi / 2, 1.5 * math.pi)]
    if ch == "C":
        w = 0.93 * h
        r = h / 2 + OVER * h
        return w, .03, .02, [A(h / 2, h / 2, r, r - s, math.radians(42), math.radians(318))]
    if ch == "U":
        w = 0.78 * h
        r = w / 2
        cy = h - r + OVER * h
        return w, .07, .07, [R(0, 0, s, cy + 1), R(w - s, 0, s, cy + 1),
                             A(r, cy, r, r - s, 0, math.pi)]
    if ch == "N":
        w = 0.78 * h
        k = s * math.hypot(w, h) / h
        return w, .07, .07, [R(0, 0, s, h), R(w - s, 0, s, h),
                             Q((0, 0), (k, 0), (w, h), (w - k, h))]
    if ch == "M":
        w = 0.94 * h
        d = 0.86 * h
        k = s * math.hypot(w / 2, d) / d
        return w, .07, .07, [R(0, 0, s, h), R(w - s, 0, s, h),
                             Q((0, 0), (k, 0), (w / 2 + k / 2, d), (w / 2 - k / 2, d)),
                             Q((w - k, 0), (w, 0), (w / 2 + k / 2, d), (w / 2 - k / 2, d))]
    if ch == "A":
        w = 0.88 * h
        k = s * math.hypot(w / 2, h) / h
        return w, .01, .01, [Q((w / 2 - k / 2, 0), (w / 2 + k / 2, 0), (k, h), (0, h)),
                             Q((w / 2 - k / 2, 0), (w / 2 + k / 2, 0), (w, h), (w - k, h)),
                             R(0.2 * w, 0.62 * h, 0.6 * w, 0.85 * s)]
    if ch == "S":
        hh = h * (1 + 2 * OVER)
        r = (hh + s) / 4
        top = -OVER * h
        return 2 * r, .03, .03, [A(r, top + r, r, r - s, math.pi / 2, math.radians(330)),
                                 A(r, top + hh - r, r, r - s, 1.5 * math.pi, math.radians(510))]
    if ch == "P":
        w = 0.64 * h
        return w, .07, .04, _bowl(h, s, w, 0.3 * h)
    if ch == "R":
        w = 0.68 * h
        b = 0.3 * h
        bw = 0.62 * h
        pieces = _bowl(h, s, bw, b)
        a = bw - b
        k = s * 1.18
        pieces.append(Q((a - 0.5 * k, 2 * b - s), (a + 0.5 * k, 2 * b - s), (w, h), (w - k, h)))
        return w, .07, .01, pieces
    raise KeyError(ch)


def word(text, h, s, x=0.0, base=0.0, tracking=0.08, align="center"):
    """Lay out a word. Returns (pieces, letter boxes, total width).

    Each piece is tagged with the index of its letter. Letter boxes are
    (x0, y0, x1, y1) in absolute coordinates.
    """
    layout = []
    pen = 0.0
    for i, ch in enumerate(text):
        w, lsb, rsb, pieces = glyph(ch, h, s)
        pen += lsb * h
        layout.append((pen, w, pieces))
        pen += w + rsb * h
        if i < len(text) - 1:
            pen += tracking * h
    total = pen
    ox = x - total / 2 if align == "center" else x
    top = base - h
    out = []
    boxes = []
    for i, (px, w, pieces) in enumerate(layout):
        for pc in pieces:
            out.append((i, translate(pc, ox + px, top)))
        boxes.append((ox + px, top, ox + px + w, base))
    return out, boxes, total


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def translate(pc, dx, dy):
    if pc[0] == "q":
        return ("q",) + tuple(v + (dx if i % 2 == 0 else dy) for i, v in enumerate(pc[1:]))
    return ("a", pc[1] + dx, pc[2] + dy) + pc[3:]


def center(pc):
    if pc[0] == "q":
        return (sum(pc[1:9:2]) / 4, sum(pc[2:9:2]) / 4)
    return (pc[1], pc[2])


def transform(pc, dx=0.0, dy=0.0, rot=0.0, scale=1.0, origin=None):
    """Rotate and scale about origin (default: piece centre), then move."""
    ox, oy = origin if origin else center(pc)
    c, s = math.cos(rot), math.sin(rot)
    if pc[0] == "q":
        out = ["q"]
        for i in range(1, 9, 2):
            x, y = (pc[i] - ox) * scale, (pc[i + 1] - oy) * scale
            out.extend((ox + x * c - y * s + dx, oy + x * s + y * c + dy))
        return tuple(out)
    x, y = (pc[1] - ox) * scale, (pc[2] - oy) * scale
    return ("a", ox + x * c - y * s + dx, oy + x * s + y * c + dy,
            pc[3] * scale, pc[4] * scale, pc[5] + rot, pc[6] + rot)


def mix(a, b, p):
    """Interpolate two pieces of the same kind."""
    return (a[0],) + tuple(lerp(u, v, p) for u, v in zip(a[1:], b[1:]))


def square(cx, cy, side, rot=0.0):
    h = side / 2
    c, s = math.cos(rot), math.sin(rot)
    pts = []
    for x, y in ((-h, -h), (h, -h), (h, h), (-h, h)):
        pts.extend((cx + x * c - y * s, cy + x * s + y * c))
    return ("q",) + tuple(pts)


def disc(cx, cy, r, a0=-math.pi / 2, sweep=TAU):
    return ("a", cx, cy, r, 0.0, a0, a0 + sweep)


def collapsed(pc):
    """A piece shrunk to nothing at its own centre (for spawning)."""
    return transform(pc, scale=0.0)


def path(ctx, pc):
    if pc[0] == "q":
        ctx.move_to(pc[1], pc[2])
        ctx.line_to(pc[3], pc[4])
        ctx.line_to(pc[5], pc[6])
        ctx.line_to(pc[7], pc[8])
        ctx.close_path()
    else:
        if pc[3] <= 0.01 or pc[6] - pc[5] <= 1e-4:
            return
        ring_seg_path(ctx, pc[1], pc[2], pc[3], max(min(pc[4], pc[3]), 0.0), pc[5], pc[6])


def match(src, dst):
    """Pair the pieces of two layouts so one can morph into the other.

    Pieces are matched within their kind, left to right. Surplus targets
    split off existing sources; surplus sources merge into a target.
    Returns a list of (source piece or None, target piece, src idx, dst idx).
    """
    pairs = []
    for kind in ("q", "a"):
        s_list = sorted([p for p in src if p[1][0] == kind], key=lambda p: center(p[1])[0])
        d_list = sorted([p for p in dst if p[1][0] == kind], key=lambda p: center(p[1])[0])
        ns, nd = len(s_list), len(d_list)
        if nd == 0:
            for sp in s_list:
                pairs.append((sp, None))
            continue
        used = set()
        for j, dp in enumerate(d_list):
            if ns == 0:
                pairs.append((None, dp))
                continue
            i = min(ns - 1, j * ns // nd)
            used.add(i)
            pairs.append((s_list[i], dp))
        for i, sp in enumerate(s_list):
            if i not in used:
                pairs.append((sp, d_list[min(nd - 1, i * nd // ns)]))
    return pairs
