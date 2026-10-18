"""Text shaping and glyph outlines.

HarfBuzz shapes each line (kerning plus Inter's case-sensitive forms for
capitals) and fontTools supplies the outlines, which are drawn as cairo
paths. Working with outlines rather than a font renderer lets the scenes
mask, crop, move and recolour individual letters.
"""

import functools
import os

import uharfbuzz as hb
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

FONT_DIR = "/usr/share/fonts/opentype/inter"


class _OpsPen(BasePen):
    """Records an outline as move / line / cubic / close operations."""

    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.ops = []

    def _moveTo(self, p):
        self.ops.append(("M", p))

    def _lineTo(self, p):
        self.ops.append(("L", p))

    def _curveToOne(self, p1, p2, p3):
        self.ops.append(("C", p1, p2, p3))

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        c1 = (p0[0] + 2 / 3 * (p1[0] - p0[0]), p0[1] + 2 / 3 * (p1[1] - p0[1]))
        c2 = (p2[0] + 2 / 3 * (p1[0] - p2[0]), p2[1] + 2 / 3 * (p1[1] - p2[1]))
        self.ops.append(("C", c1, c2, p2))

    def _closePath(self):
        self.ops.append(("Z",))

    def _endPath(self):
        self.ops.append(("Z",))


def _ops_bounds(ops):
    xs, ys = [], []
    for op in ops:
        for p in op[1:]:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


class Font:
    def __init__(self, filename):
        path = os.path.join(FONT_DIR, filename)
        self.face = hb.Face(hb.Blob.from_file_path(path))
        self.hb = hb.Font(self.face)
        self.upem = self.face.upem
        tt = TTFont(path)
        self.glyph_set = tt.getGlyphSet()
        self.order = tt.getGlyphOrder()
        self.cap_height = tt["OS/2"].sCapHeight / self.upem
        self.x_height = tt["OS/2"].sxHeight / self.upem
        self._ops = {}

    def ops(self, gid):
        if gid not in self._ops:
            pen = _OpsPen(self.glyph_set)
            self.glyph_set[self.order[gid]].draw(pen)
            self._ops[gid] = (pen.ops, _ops_bounds(pen.ops))
        return self._ops[gid]


@functools.lru_cache(maxsize=None)
def font(name):
    return Font(name)


DISPLAY_BOLD = "InterDisplay-Bold.otf"
DISPLAY_BLACK = "InterDisplay-Black.otf"
DISPLAY_SEMI = "InterDisplay-SemiBold.otf"
DISPLAY_MEDIUM = "InterDisplay-Medium.otf"
DISPLAY_REGULAR = "InterDisplay-Regular.otf"
DISPLAY_LIGHT = "InterDisplay-Light.otf"
DISPLAY_THIN = "InterDisplay-Thin.otf"
TEXT_MEDIUM = "Inter-Medium.otf"
TEXT_SEMI = "Inter-SemiBold.otf"
TEXT_REGULAR = "Inter-Regular.otf"


class Glyph:
    __slots__ = ("ops", "x", "adv", "bounds", "char")

    def __init__(self, ops, x, adv, bounds, char):
        self.ops = ops
        self.x = x          # pen position, px from the line origin
        self.adv = adv      # advance in px (including tracking)
        self.bounds = bounds  # (x0, y0, x1, y1) in px, y down, origin baseline
        self.char = char


class TextLine:
    """A shaped single line of text at a given size (px)."""

    def __init__(self, fontname, text, size, tracking=0.0, caps=True):
        f = font(fontname)
        self.font = f
        self.text = text
        self.size = size
        self.scale = size / f.upem
        self.cap = f.cap_height * size
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        features = {"kern": True, "liga": True, "calt": True, "case": caps}
        hb.shape(f.hb, buf, features)
        s = self.scale
        track = tracking * size
        pen_x = 0.0
        self.glyphs = []
        n = len(buf.glyph_infos)
        for i, (info, pos) in enumerate(zip(buf.glyph_infos, buf.glyph_positions)):
            ops, (bx0, by0, bx1, by1) = f.ops(info.codepoint)
            adv = pos.x_advance * s + (track if i < n - 1 else 0.0)
            gx = pen_x + pos.x_offset * s
            bounds = (gx + bx0 * s, -by1 * s, gx + bx1 * s, -by0 * s)
            ch = text[info.cluster] if info.cluster < len(text) else ""
            self.glyphs.append(Glyph(ops, gx, adv, bounds, ch))
            pen_x += adv
        self.width = pen_x
        ink = [g.bounds for g in self.glyphs if g.ops]
        if ink:
            self.ink = (min(b[0] for b in ink), min(b[1] for b in ink),
                        max(b[2] for b in ink), max(b[3] for b in ink))
        else:
            self.ink = (0, 0, 0, 0)

    @property
    def ink_width(self):
        return self.ink[2] - self.ink[0]

    def glyph_path(self, ctx, i, x, y):
        """Append glyph i's outline, line origin at (x, y) on the baseline."""
        g = self.glyphs[i]
        s = self.scale
        ox = x + g.x
        for op in g.ops:
            k = op[0]
            if k == "M":
                ctx.move_to(ox + op[1][0] * s, y - op[1][1] * s)
            elif k == "L":
                ctx.line_to(ox + op[1][0] * s, y - op[1][1] * s)
            elif k == "C":
                (a, b), (c, d), (e, f) = op[1], op[2], op[3]
                ctx.curve_to(ox + a * s, y - b * s, ox + c * s, y - d * s,
                             ox + e * s, y - f * s)
            else:
                ctx.close_path()

    def path(self, ctx, x, y):
        for i in range(len(self.glyphs)):
            self.glyph_path(ctx, i, x, y)

    def draw(self, ctx, x, y, color, alpha=1.0):
        from .core import set_color
        self.path(ctx, x, y)
        set_color(ctx, color, alpha)
        ctx.fill()

    def x_for(self, x, align):
        """Line origin x for a given anchor x and alignment (ink based)."""
        if align == "left":
            return x - self.ink[0]
        if align == "right":
            return x - self.ink[2]
        return x - (self.ink[0] + self.ink[2]) / 2


@functools.lru_cache(maxsize=512)
def line(fontname, text, size, tracking=0.0, caps=True):
    return TextLine(fontname, text, size, tracking, caps)


def fit_size(fontname, text, width, tracking=0.0, caps=True):
    """Size (px) at which the ink of text spans exactly `width` pixels."""
    probe = TextLine(fontname, text, 1000.0, tracking, caps)
    return 1000.0 * width / probe.ink_width
