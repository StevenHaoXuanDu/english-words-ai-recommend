"""Low-level drawing for render.py: colours, stacking layout, rasteriser, glow, text.

Everything here works in 1920x1080 "design" pixels; a Raster built with scale s
multiplies every coordinate by s, so previews render natively at lower resolution.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
Y0 = 500.0                      # row centreline
RECT_W, RECT_H = 10.0, 28.0
COL_PITCH, ROW_PITCH = 14.0, 32.0
GROUP_TOL = 0.0015              # opinions closer than this stack into one block
ROW_L = 160.0 - RECT_W / 2 - 8  # bracket clip: 8 px beyond the first / last rect
ROW_R = 1760.0 + RECT_W / 2 + 8

GLOW_SIGMA_IN, GLOW_GAIN_IN = 6.0, 0.55     # "blur 12 px, 35 %" from the script, lifted
GLOW_SIGMA_OUT, GLOW_GAIN_OUT = 26.0, 0.24  # wide faint bloom


def X(x):
    """Opinion -> design-pixel x."""
    return 160.0 + 1600.0 * np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------- colour
def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def srgb_to_oklab(c):
    return np.cbrt(_to_lin(c) @ _M1.T) @ _M2.T


def oklab_to_srgb(lab):
    lms = (lab @ np.linalg.inv(_M2).T) ** 3
    return _to_srgb(lms @ np.linalg.inv(_M1).T)


TEAL, GOLD, VERM = hex_rgb("#5BB5A8"), hex_rgb("#E8C066"), hex_rgb("#D4573B")
WHITE_GOLD = hex_rgb("#FFF4D6")
CAPTION_RGB = hex_rgb("#F3E9D2")

_LUT_N = 2049
_t = np.linspace(0.0, 1.0, _LUT_N)
_lt, _lg, _lv = srgb_to_oklab(TEAL), srgb_to_oklab(GOLD), srgb_to_oklab(VERM)
_lab = np.where((_t <= 0.5)[:, None],
                _lt + (_lg - _lt) * np.clip(_t / 0.5, 0, 1)[:, None],
                _lg + (_lv - _lg) * np.clip((_t - 0.5) / 0.5, 0, 1)[:, None])
LUT = oklab_to_srgb(_lab)
LUT[0], LUT[(_LUT_N - 1) // 2], LUT[-1] = TEAL, GOLD, VERM


def opinion_rgb(x):
    i = np.clip(np.rint(np.asarray(x) * (_LUT_N - 1)).astype(int), 0, _LUT_N - 1)
    return LUT[i]


# 8-bit sRGB <-> linear light, for averaging motion-blur sub-frames
SRGB_TO_LIN = _to_lin(np.arange(256) / 255.0).astype(np.float32)
_LIN_N = 16384
LIN_TO_SRGB = np.clip(np.rint(_to_srgb(np.linspace(0.0, 1.0, _LIN_N)) * 255.0), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- layout
class Layout:
    """Screen layout of one opinion vector: groups stack into centred column-major grids."""

    def __init__(self, op):
        op = np.asarray(op, dtype=np.float64)
        self.op = op
        n = len(op)
        order = np.argsort(op, kind="stable")
        groups, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if op[b] - op[a] <= GROUP_TOL:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        self.wx = np.empty(n)
        self.wy = np.empty(n)
        self.gsize = np.empty(n, dtype=int)   # size of the group each agent stacks in
        self.groups = []          # (centre x, half width, half height, members)
        for g in groups:
            g = sorted(int(i) for i in g)
            k = len(g)
            self.gsize[g] = k
            rows, cols = min(10, k), math.ceil(k / 10)
            xm = float(X(op[g].mean()))
            for s, i in enumerate(g):
                c, r = divmod(s, 10)
                self.wx[i] = xm + (c - (cols - 1) / 2) * COL_PITCH
                self.wy[i] = Y0 + ((rows - 1) / 2 - r) * ROW_PITCH
            self.groups.append((xm, (cols - 1) / 2 * COL_PITCH + RECT_W / 2,
                                (rows - 1) / 2 * ROW_PITCH + RECT_H / 2, g))


# ----------------------------------------------------------------------------- scene
class Scene:
    """Primitives in design pixels (after camera). Raster draws them."""

    def __init__(self):
        self.rects = []      # dict(x, y, w, h, rgb(n,3), level(n), glow(n))
        self.lines = []      # (x0, y0, x1, y1, width, rgb, alpha, glow)
        self.dots = []       # (x, y, radius, rgb, alpha, glow)
        self.zoom = 1.0      # scales glow radius
        self.glow_gain = 1.0
        self.texts = []      # (kind, text, opacity)

    def add_rects(self, x, y, w, h, rgb, level, glow):
        n = len(x)
        self.rects.append(dict(x=np.asarray(x, float), y=np.asarray(y, float),
                               w=np.broadcast_to(np.asarray(w, float), (n,)),
                               h=np.broadcast_to(np.asarray(h, float), (n,)),
                               rgb=np.asarray(rgb, float).reshape(n, 3),
                               level=np.broadcast_to(np.asarray(level, float), (n,)),
                               glow=np.broadcast_to(np.asarray(glow, float), (n,))))

    def line(self, x0, y0, x1, y1, width, rgb, alpha, glow=0.0):
        if alpha > 1e-3:
            self.lines.append((x0, y0, x1, y1, width, np.asarray(rgb, float), alpha, glow))

    def dot(self, x, y, r, rgb, alpha, glow=1.0):
        if alpha > 1e-3:
            self.dots.append((x, y, r, np.asarray(rgb, float), alpha, glow))

    def text(self, kind, text, opacity):
        if opacity > 1e-3:
            self.texts.append((kind, text, float(min(1.0, opacity))))


# ----------------------------------------------------------------------------- raster
def _S(x):
    """Integral of clamp(x, 0, 1)."""
    return np.where(x <= 0.0, 0.0, np.where(x >= 1.0, x - 0.5, 0.5 * x * x))


def _span(a, b, n, h=0.0):
    """Pixel range [i0, i1) and per-pixel coverage of the interval [a, b], clipped to [0, n).
    h > 0: time-averaged coverage while the interval slides uniformly over [-h, +h]
    (analytic box-blur along the motion = exact motion blur for linear motion)."""
    i0, i1 = max(int(math.floor(a - h)), 0), min(int(math.ceil(b + h)), n)
    if i1 <= i0:
        return None
    p = np.arange(i0, i1, dtype=np.float32)
    if h < 1e-3:
        c = np.minimum(p + 1.0, b) - np.maximum(p, a)
    else:
        c = (_S(b - p + h) - _S(b - p - h) - _S(a - p + h) + _S(a - p - h)) / (2.0 * h)
    return i0, i1, np.clip(c, 0.0, 1.0).astype(np.float32)


def _box_cov(x0, y0, x1, y1, w, h, hx=0.0, hy=0.0):
    sx, sy = _span(x0, x1, w, hx), _span(y0, y1, h, hy)
    if sx is None or sy is None:
        return None
    return sx[0], sx[1], sy[0], sy[1], sy[2][:, None] * sx[2][None, :]


def _seg_cov(x0, y0, x1, y1, width, w, h):
    """Anti-aliased coverage of a segment of the given width (distance field)."""
    hw = width / 2.0
    pad = hw + 1.5
    i0, i1 = max(int(math.floor(min(x0, x1) - pad)), 0), min(int(math.ceil(max(x0, x1) + pad)), w)
    j0, j1 = max(int(math.floor(min(y0, y1) - pad)), 0), min(int(math.ceil(max(y0, y1) + pad)), h)
    if i1 <= i0 or j1 <= j0:
        return None
    px = np.arange(i0, i1, dtype=np.float32)[None, :] + 0.5
    py = np.arange(j0, j1, dtype=np.float32)[:, None] + 0.5
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        d = np.hypot(px - x0, py - y0)
    else:
        u = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0)
        d = np.hypot(px - (x0 + u * dx), py - (y0 + u * dy))
    if width >= 1.0:
        cov = np.clip(hw + 0.5 - d, 0.0, 1.0)
    else:
        cov = width * np.clip(1.0 - d, 0.0, 1.0)
    return i0, i1, j0, j1, cov


def _next_fast(n):
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


class Raster:
    def __init__(self, scale=1.0):
        self.s = scale
        self.W = int(round(W * scale / 2)) * 2
        self.H = int(round(H * scale / 2)) * 2
        self.q = 4 if scale >= 0.75 else (2 if scale >= 0.4 else 1)  # glow downsample
        self.text = TextCache(scale, self.W, self.H)

    # -- one frame ------------------------------------------------------------------
    def render(self, sc, texts=True):
        s, q = self.s, self.q
        frame = np.zeros((self.H, self.W, 3), np.uint8)
        rects = []
        for b in sc.rects:
            lv = b["level"]
            smx, smy = b.get("sx"), b.get("sy")      # motion-blur smear (design px), optional
            for i in np.argsort(lv, kind="stable"):      # bright ones drawn last (on top)
                if lv[i] <= 1e-3:
                    continue
                hw, hh = b["w"][i] * s / 2, b["h"][i] * s / 2
                x, y = b["x"][i] * s, b["y"][i] * s
                hx = abs(float(smx[i])) * s / 2 if smx is not None else 0.0
                hy = abs(float(smy[i])) * s / 2 if smy is not None else 0.0
                rects.append((x - hw, y - hh, x + hw, y + hh, b["rgb"][i], lv[i], b["glow"][i], hx, hy))
        sm = getattr(sc, "line_smear", None)
        lines = [(l[0] * s, l[1] * s, l[2] * s, l[3] * s, max(l[4] * s, 0.35)) + l[5:8]
                 + ((abs(sm[k][0]) * s / 2, abs(sm[k][1]) * s / 2) if sm else (0.0, 0.0))
                 for k, l in enumerate(sc.lines)]
        dots = [(d[0] * s, d[1] * s, max(d[2] * s, 0.6)) + d[3:] for d in sc.dots]

        if rects or lines or dots:
            sig_in = GLOW_SIGMA_IN * sc.zoom * s
            sig_out = GLOW_SIGMA_OUT * sc.zoom * s
            margin = 3.2 * sig_out + 4
            xs0 = ([r[0] - r[7] for r in rects] + [min(l[0], l[2]) - l[4] - l[8] for l in lines]
                   + [d[0] - d[2] for d in dots])
            xs1 = ([r[2] + r[7] for r in rects] + [max(l[0], l[2]) + l[4] + l[8] for l in lines]
                   + [d[0] + d[2] for d in dots])
            ys0 = ([r[1] - r[8] for r in rects] + [min(l[1], l[3]) - l[4] - l[9] for l in lines]
                   + [d[1] - d[2] for d in dots])
            ys1 = ([r[3] + r[8] for r in rects] + [max(l[1], l[3]) + l[4] + l[9] for l in lines]
                   + [d[1] + d[2] for d in dots])
            bx0 = max(0, int(math.floor(min(xs0) - margin)) // q * q)
            by0 = max(0, int(math.floor(min(ys0) - margin)) // q * q)
            bx1 = min(self.W, -(-int(math.ceil(max(xs1) + margin)) // q) * q)
            by1 = min(self.H, -(-int(math.ceil(max(ys1) + margin)) // q) * q)
            if bx1 > bx0 and by1 > by0:
                self._composite(frame, sc, rects, lines, dots, (bx0, by0, bx1, by1), sig_in, sig_out)

        if texts:
            self.draw_texts(frame, sc)
        return frame

    def draw_texts(self, frame, sc):
        for kind, txt, op in sc.texts:
            self.text.draw(frame, kind, txt, op)

    def render_blurred(self, scenes, bounds, text_scene):
        """Motion blur over the shutter interval. scenes[k] is the scene at the centre of
        sub-interval k and bounds[k], bounds[k+1] the scenes at its ends: each rect (and each
        axis-aligned line) of scenes[k] is box-blurred along its own displacement over the
        sub-interval, so the sub-frames tile the motion without ripple. Sub-frames are
        averaged in linear light; the text of `text_scene` is drawn once on top."""
        for k, sc in enumerate(scenes):
            A, B = bounds[k], bounds[k + 1]
            for bc, ba, bb in zip(sc.rects, A.rects, B.rects):
                if len(ba["x"]) == len(bb["x"]) == len(bc["x"]):
                    bc["sx"], bc["sy"] = bb["x"] - ba["x"], bb["y"] - ba["y"]
            if len(A.lines) == len(B.lines) == len(sc.lines):
                sc.line_smear = [(0.5 * (lb[0] - la[0] + lb[2] - la[2]), 0.5 * (lb[1] - la[1] + lb[3] - la[3]))
                                 for la, lb in zip(A.lines, B.lines)]
        if len(scenes) == 1:
            frame = self.render(scenes[0], texts=False)
            self.draw_texts(frame, text_scene)
            return frame
        acc = np.zeros((self.H, self.W, 3), np.float32)
        r0, r1 = self.H, 0
        for sc in scenes:
            f = self.render(sc, texts=False)
            rows = np.flatnonzero(f.reshape(self.H, -1).any(1))
            if rows.size:
                a, b = int(rows[0]), int(rows[-1]) + 1
                acc[a:b] += SRGB_TO_LIN[f[a:b]]
                r0, r1 = min(r0, a), max(r1, b)
        frame = np.zeros((self.H, self.W, 3), np.uint8)
        if r1 > r0:
            lin = acc[r0:r1] * (1.0 / len(scenes))
            idx = (np.clip(lin, 0.0, 1.0) * (_LIN_N - 1) + 0.5).astype(np.int32)
            frame[r0:r1] = LIN_TO_SRGB[idx]
        self.draw_texts(frame, text_scene)
        return frame

    def _composite(self, frame, sc, rects, lines, dots, box, sig_in, sig_out):
        q = self.q
        bx0, by0, bx1, by1 = box
        bw, bh = bx1 - bx0, by1 - by0
        rgb = np.zeros((bh, bw, 3), np.float32)
        alpha = np.zeros((bh, bw), np.float32)

        # glow source, downsampled by q, padded so the FFT blur does not wrap around
        pad = int(math.ceil(3.2 * sig_out / q)) + 2
        gh, gw = _next_fast(bh // q + 2 * pad), _next_fast(bw // q + 2 * pad)
        g = np.zeros((gh, gw, 3), np.float32)
        gx0, gy0 = bx0 / q - pad, by0 / q - pad        # glow-buffer origin in q-pixels

        for x0, y0, x1, y1, col, lv, gl, hx, hy in rects:
            c = _box_cov(x0 - bx0, y0 - by0, x1 - bx0, y1 - by0, bw, bh, hx, hy)
            if c is not None:
                i0, i1, j0, j1, cov = c
                a = cov * min(lv, 1.0)
                reg = rgb[j0:j1, i0:i1]
                reg *= (1.0 - a)[..., None]
                reg += (cov * lv)[..., None] * col.astype(np.float32)
                alpha[j0:j1, i0:i1] += (1.0 - alpha[j0:j1, i0:i1]) * a
            if gl > 0:
                c = _box_cov(x0 / q - gx0, y0 / q - gy0, x1 / q - gx0, y1 / q - gy0, gw, gh, hx / q, hy / q)
                if c is not None:
                    i0, i1, j0, j1, cov = c
                    g[j0:j1, i0:i1] += (cov * (lv * gl))[..., None] * col.astype(np.float32)

        for x0, y0, x1, y1, wd, col, al, gl, hx, hy in lines:
            self._line(rgb, alpha, g, (x0 - bx0, y0 - by0, x1 - bx0, y1 - by0), wd, col, al, bw, bh, False,
                       hx, hy)
            if gl > 0:
                self._line(None, None, g, (x0 / q - gx0, y0 / q - gy0, x1 / q - gx0, y1 / q - gy0),
                           max(wd / q, 0.35), col, al * gl, gw, gh, True, hx / q, hy / q)

        for x, y, r, col, al, gl in dots:
            c = _seg_cov(x - bx0, y - by0, x - bx0, y - by0, 2 * r, bw, bh)
            if c is not None:
                i0, i1, j0, j1, cov = c
                a = cov * al
                rgb[j0:j1, i0:i1] = np.maximum(rgb[j0:j1, i0:i1], a[..., None] * col.astype(np.float32))
                alpha[j0:j1, i0:i1] = np.maximum(alpha[j0:j1, i0:i1], a)
            if gl > 0:
                c = _seg_cov(x / q - gx0, y / q - gy0, x / q - gx0, y / q - gy0, max(2 * r / q, 0.5), gw, gh)
                if c is not None:
                    i0, i1, j0, j1, cov = c
                    g[j0:j1, i0:i1] += (cov * al * gl)[..., None] * col.astype(np.float32)

        # blur the glow source with an analytic two-gaussian kernel in the frequency domain
        fy = np.fft.fftfreq(gh)[:, None]
        fx = np.fft.rfftfreq(gw)[None, :]
        r2 = (fx * fx + fy * fy) * (2 * math.pi ** 2)
        si, so = sig_in / q, sig_out / q
        k = (GLOW_GAIN_IN * np.exp(-r2 * si * si) + GLOW_GAIN_OUT * np.exp(-r2 * so * so)) * sc.glow_gain
        G = np.fft.irfft2(np.fft.rfft2(g, axes=(0, 1)) * k[..., None].astype(np.complex64),
                          s=(gh, gw), axes=(0, 1)).astype(np.float32)
        if q > 1:
            box = (pad, pad, pad + bw / q, pad + bh / q)
            up = np.empty((bh, bw, 3), np.float32)
            for ch in range(3):
                im = Image.fromarray(np.ascontiguousarray(G[:, :, ch]), mode="F")
                up[:, :, ch] = np.asarray(im.resize((bw, bh), Image.BILINEAR, box=box))
        else:
            up = G[pad:pad + bh, pad:pad + bw]
        np.maximum(up, 0.0, out=up)
        up *= (1.0 - alpha)[..., None]
        rgb += up
        np.clip(rgb, 0.0, 1.0, out=rgb)
        rgb *= 255.0
        rgb += 0.5
        frame[by0:by1, bx0:bx1] = rgb.astype(np.uint8)

    @staticmethod
    def _line(rgb, alpha, g, seg, wd, col, al, w, h, additive, hx=0.0, hy=0.0):
        x0, y0, x1, y1 = seg
        if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:   # axis aligned -> crisp box (motion-blurred)
            hw = wd / 2.0
            if abs(x1 - x0) < 1e-6:
                c = _box_cov(x0 - hw, min(y0, y1), x0 + hw, max(y0, y1), w, h, hx, hy)
            else:
                c = _box_cov(min(x0, x1), y0 - hw, max(x0, x1), y0 + hw, w, h, hx, hy)
        else:
            c = _seg_cov(x0, y0, x1, y1, wd, w, h)
        if c is None:
            return
        i0, i1, j0, j1, cov = c
        a = cov * al
        if additive:
            g[j0:j1, i0:i1] += a[..., None] * col.astype(np.float32)
        else:
            rgb[j0:j1, i0:i1] = np.maximum(rgb[j0:j1, i0:i1], a[..., None] * col.astype(np.float32))
            alpha[j0:j1, i0:i1] = np.maximum(alpha[j0:j1, i0:i1], a)


# ----------------------------------------------------------------------------- text
SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc"
SERIF_INDEX = 2   # Noto Serif CJK SC
MONO_CHAIN = [
    "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf",
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

CAPTION_PX, CAPTION_Y = 48, 850
ANNOT_PX, ANNOT_BASELINE, ANNOT_TRACK, ANNOT_ALPHA = 18, 1080 - 48, 0.15, 0.45
CARD_STYLE = {"card_s04": (64, 380), "card_s22": (56, 612)}   # px size, centre y


_CARD_BOX = {}


def card_box(kind, text):
    """Ink box (x0, y0, x1, y1) of a title card in design pixels."""
    key = (kind, text)
    if key not in _CARD_BOX:
        px, cy = CARD_STYLE[kind]
        l, t, r, b = ImageFont.truetype(SERIF, px, index=SERIF_INDEX).getbbox(text, anchor="mm")
        _CARD_BOX[key] = (W / 2 + l, cy + t, W / 2 + r, cy + b)
    return _CARD_BOX[key]


def has_glyph(font, ch):
    """True unless the font would draw its .notdef box for ch."""
    if ch.isspace():
        return True
    a, b = font.getmask(ch), font.getmask("\U000FFFFD")
    return not (a.size == b.size and bytes(a) == bytes(b))


class TextCache:
    def __init__(self, scale, w, h):
        self.s, self.W, self.H = scale, w, h
        self.cache = {}
        self._serif = {}
        self.mono = [ImageFont.truetype(p, max(1, round(ANNOT_PX * scale))) for p in MONO_CHAIN]
        self.mono_cell = self.mono[0].getlength("M") + ANNOT_TRACK * ANNOT_PX * scale
        self._pick = {}

    def serif(self, px):
        px = max(1, round(px * self.s))
        if px not in self._serif:
            self._serif[px] = ImageFont.truetype(SERIF, px, index=SERIF_INDEX)
        return self._serif[px]

    def mono_font(self, ch):
        if ch not in self._pick:
            for f in self.mono:
                if has_glyph(f, ch):
                    self._pick[ch] = f
                    break
            else:
                raise ValueError(f"no font has a glyph for {ch!r} (U+{ord(ch):04X})")
        return self._pick[ch]

    def _centred(self, text, px, cy):
        f = self.serif(px)
        l, t, r, b = f.getbbox(text, anchor="mm")
        ax, ay = self.W / 2, cy * self.s
        x0, y0 = int(math.floor(ax + l)) - 2, int(math.floor(ay + t)) - 2
        im = Image.new("L", (int(r - l) + 6, int(b - t) + 6), 0)
        ImageDraw.Draw(im).text((ax - x0, ay - y0), text, font=f, fill=255, anchor="mm")
        return np.asarray(im), x0, y0, CAPTION_RGB

    def _annotation(self, text):
        s = self.s
        n = len(text)
        total = n * self.mono_cell - ANNOT_TRACK * ANNOT_PX * s
        x = self.W / 2 - total / 2
        base = ANNOT_BASELINE * s
        hgt = int(ANNOT_PX * s * 2.2) + 4
        x0, y0 = int(math.floor(x)) - 4, int(math.floor(base - ANNOT_PX * s * 1.5)) - 2
        im = Image.new("L", (int(total) + 12, hgt), 0)
        d = ImageDraw.Draw(im)
        cellw = self.mono[0].getlength("M")
        for k, ch in enumerate(text):
            if ch == " ":
                continue
            f = self.mono_font(ch)
            gx = x + k * self.mono_cell + (cellw - f.getlength(ch)) / 2 - x0
            d.text((gx, base - y0), ch, font=f, fill=255, anchor="ls")
        return np.asarray(im), x0, y0, GOLD

    def get(self, kind, text):
        key = (kind, text)
        if key not in self.cache:
            if kind == "caption":
                self.cache[key] = self._centred(text, CAPTION_PX, CAPTION_Y)
            elif kind == "annotation":
                self.cache[key] = self._annotation(text)
            else:
                px, cy = CARD_STYLE[kind]
                self.cache[key] = self._centred(text, px, cy)
            if len(self.cache) > 400:          # counters create many strings
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def draw(self, frame, kind, text, opacity):
        mask, x0, y0, col = self.get(kind, text)
        if kind == "annotation":
            opacity *= ANNOT_ALPHA
        h, w = mask.shape
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
        if fx1 <= fx0 or fy1 <= fy0:
            return
        m = mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32) * (opacity / 255.0)
        reg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        reg += (col.astype(np.float32) * 255.0 - reg) * m[..., None]
        frame[fy0:fy1, fx0:fx1] = (reg + 0.5).astype(np.uint8)
