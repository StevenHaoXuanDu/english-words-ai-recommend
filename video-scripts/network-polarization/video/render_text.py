"""Animated typography for render.py, drawn last, straight onto the finished frame.

Scene.texts holds (kind, text, opacity, params). params carry the animation clock:
tin = seconds since the element appeared, tout = seconds until it disappears.

    caption      serif line; characters rise and fade in on a fast stagger, the line leaves
                 with a quick fade + blur. Fully readable from tin = REVEAL_MAX on.
    annotation   small mono line at the bottom; types on with a block cursor that blinks a
                 few times once the line is complete, fades out.
    card_s04/22  serif title card; per-character focus-in while the tracking settles.
    chapter      ("01 · 村子", "THE VILLAGE") top-left label with a drawn hairline.
    hud          top-right mono readout + optional sparkline (params["spark"]).
    mono_at      small mono text at a design-px position (axis labels).
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from render_core import (ANNOT_ALPHA, ANNOT_BASELINE, ANNOT_PX, ANNOT_TRACK, CAPTION_PX, CAPTION_RGB,
                         CAPTION_Y, CARD_STYLE, GOLD, W, TextCache, _seg_cov)

# caption reveal: char k starts at k * stagger; the whole line is in by REVEAL_MAX
CAP_FADE, CAP_STAG_MAX, CAP_STAG_TOTAL, CAP_RISE = 0.16, 0.025, 0.16, 8.0
REVEAL_MAX = CAP_STAG_TOTAL + CAP_FADE
CAP_EXIT, CAP_EXIT_BLUR = 0.22, 3.5
TYPE_CPS, CURSOR_HOLD, CURSOR_PERIOD = 55.0, 0.95, 0.5
ANNOT_EXIT = 0.25
CARD_FADE, CARD_TRACK, CARD_SETTLE, CARD_EXIT = 0.55, 0.16, 1.3, 0.45
CHAPTER_X, CHAPTER_BASE, CHAPTER_PX = 120.0, 104.0, 30
CHAPTER_MONO_PX, CHAPTER_MONO_BASE, CHAPTER_LINE_Y = 13, 146.0, 121.0
CHAPTER_ALPHA, CHAPTER_EXIT = 0.82, 0.4
HUD_RIGHT, HUD_BASE, HUD_PX, HUD_ALPHA = 1800.0, 92.0, 14, 0.42
SPARK_W, SPARK_H, SPARK_TOP = 150.0, 26.0, 108.0


def ease_out(p):
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


def ease_out_expo(p):
    p = min(1.0, max(0.0, p))
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def blend(frame, mask, x0, y0, col, opacity):
    """Alpha-blend a uint8/float mask (0..255) of colour col into the frame at (x0, y0)."""
    if opacity <= 1e-3:
        return
    h, w = mask.shape
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return
    m = mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32) * (opacity / 255.0)
    reg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    reg += (np.asarray(col, np.float32) * 255.0 - reg) * m[..., None]
    frame[fy0:fy1, fx0:fx1] = (reg + 0.5).astype(np.uint8)


class TextFX:
    def __init__(self, scale, w, h):
        self.s, self.W, self.H = scale, w, h
        self.base = TextCache(scale, w, h)
        self.cache = {}
        self._mono = {}

    def _get(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
            if len(self.cache) > 600:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    # -- serif glyph runs -------------------------------------------------------------
    def glyphs(self, text, px, cx=None, cy=0.0, left=None):
        """Per-character masks of a serif line: list of (mask, x0, y0, centre x) in raster px.
        Centred on design x cx (default: frame centre) or starting at design x `left`;
        vertical middle (PIL anchor 'm') at design y cy."""
        def build():
            s = self.s
            f = self.base.serif(px)
            total = f.getlength(text)
            ay = cy * s
            xs = (left * s) if left is not None else ((cx * s if cx is not None else self.W / 2) - total / 2)
            asc, desc = f.getmetrics()
            hgt = int(asc + desc + 8)
            pad = int(math.ceil(px * s * 0.25)) + 2
            out = []
            for k, ch in enumerate(text):
                if ch.isspace():
                    continue
                xk = xs + f.getlength(text[:k])
                adv = f.getlength(ch)
                x0 = int(math.floor(xk)) - pad
                y0 = int(math.floor(ay - hgt / 2))
                im = Image.new("L", (int(math.ceil(adv)) + 2 * pad + 2, hgt), 0)
                ImageDraw.Draw(im).text((xk - x0, ay - y0), ch, font=f, fill=255, anchor="lm")
                out.append((np.asarray(im), x0, y0, xk + adv / 2))
            return out
        return self._get(("glyphs", text, px, cx, cy, left), build)

    def line_mask(self, glyphs):
        """Union of a glyph run as one mask (x0, y0, mask)."""
        x0 = min(g[1] for g in glyphs)
        y0 = min(g[2] for g in glyphs)
        x1 = max(g[1] + g[0].shape[1] for g in glyphs)
        y1 = max(g[2] + g[0].shape[0] for g in glyphs)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for g, gx, gy, _ in glyphs:
            sub = m[gy - y0:gy - y0 + g.shape[0], gx - x0:gx - x0 + g.shape[1]]
            np.maximum(sub, g, out=sub)
        return x0, y0, m

    def blurred(self, mask, r):
        if r < 0.3:
            return mask, 0
        pad = int(math.ceil(3 * r))
        im = Image.fromarray(np.pad(mask, pad))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(r))), pad

    # -- mono ---------------------------------------------------------------------------
    def mono(self, px):
        key = max(1, round(px * self.s))
        if key not in self._mono:
            from render_core import MONO_CHAIN
            self._mono[key] = [ImageFont.truetype(p, key) for p in MONO_CHAIN]
        return self._mono[key]

    def mono_line(self, text, px, track, x, base, align):
        """Mask of a tracked mono line. align: 'c' centred on x, 'l' from x, 'r' ending at x.
        Returns (mask, x0, y0, cell px, first-cell raster x)."""
        def build():
            s = self.s
            fonts = self.mono(px)
            cellw = fonts[0].getlength("M")
            cell = cellw + track * px * s
            n = len(text)
            total = n * cell - track * px * s
            if align == "c":
                xs = x * s - total / 2
            elif align == "r":
                xs = x * s - total
            else:
                xs = x * s
            b = base * s
            hgt = int(px * s * 2.2) + 4
            x0, y0 = int(math.floor(xs)) - 4, int(math.floor(b - px * s * 1.5)) - 2
            im = Image.new("L", (int(total) + 12, hgt), 0)
            d = ImageDraw.Draw(im)
            for k, ch in enumerate(text):
                if ch == " ":
                    continue
                f = self.base.mono_font(ch) if px == ANNOT_PX else self._pick(fonts, ch)
                gx = xs + k * cell + (cellw - f.getlength(ch)) / 2 - x0
                d.text((gx, b - y0), ch, font=f, fill=255, anchor="ls")
            return np.asarray(im), x0, y0, cell, xs
        return self._get(("mono", text, px, track, x, base, align), build)

    @staticmethod
    def _pick(fonts, ch):
        from render_core import has_glyph
        for f in fonts:
            if has_glyph(f, ch):
                return f
        raise ValueError(f"no font has a glyph for {ch!r}")

    def typed(self, frame, text, px, track, x, base, align, col, op, tin, cps=TYPE_CPS, cursor=True):
        """Type-on of a mono line with a block cursor."""
        mask, x0, y0, cell, xs = self.mono_line(text, px, track, x, base, align)
        n = len(text)
        k = n if cps <= 0 else min(n, int(tin * cps))
        if k <= 0 and not cursor:
            return
        if k > 0:
            cut = int(round(xs + k * cell - x0))
            blend(frame, mask[:, :max(0, min(cut, mask.shape[1]))], x0, y0, col, op)
        if not cursor:
            return
        t_done = n / cps if cps > 0 else 0.0
        if tin < t_done:
            c = 1.0
        elif tin < t_done + CURSOR_HOLD:
            c = 0.5 + 0.5 * math.cos(2 * math.pi * (tin - t_done) / CURSOR_PERIOD)
            c *= 1 - max(0.0, (tin - t_done - CURSOR_HOLD + 0.25) / 0.25)
        else:
            return
        s = self.s
        cx0 = xs + min(k, n) * cell
        cw = max(1, int(round(0.62 * px * s)))
        ch = max(1, int(round(0.95 * px * s)))
        b = int(round(base * s))
        blk = np.full((ch, cw), 255, np.uint8)
        blend(frame, blk, int(round(cx0)), b - ch + int(round(0.12 * px * s)), col, op * 0.85 * c)

    # -- elements ----------------------------------------------------------------------
    def caption(self, frame, text, op, tin, tout):
        g = self.glyphs(text, CAPTION_PX, cy=CAPTION_Y)
        n = len(g)
        stag = min(CAP_STAG_MAX, CAP_STAG_TOTAL / max(n - 1, 1))
        ex = 1 - min(1.0, max(0.0, tout / CAP_EXIT))
        s = self.s
        if ex > 0:                                  # leaving: whole line, fade + blur + drift up
            x0, y0, m = self._get(("line", text), lambda: self.line_mask(g))
            r = CAP_EXIT_BLUR * s * ex
            if r >= 0.3:
                m, pad = self.blurred(m, r)
                x0, y0 = x0 - pad, y0 - pad
            blend(frame, m, x0, y0 - int(round(4 * s * ex)), CAPTION_RGB, op * (1 - ex) ** 1.3)
            return
        if tin >= (n - 1) * stag + CAP_FADE:
            x0, y0, m = self._get(("line", text), lambda: self.line_mask(g))
            blend(frame, m, x0, y0, CAPTION_RGB, op)
            return
        for k, (m, x0, y0, _) in enumerate(g):
            p = ease_out((tin - k * stag) / CAP_FADE)
            if p <= 0:
                break
            blend(frame, m, x0, y0 + int(round(CAP_RISE * s * (1 - p))), CAPTION_RGB, op * p)

    def annotation(self, frame, text, op, tin, tout):
        ex = 1 - min(1.0, max(0.0, tout / ANNOT_EXIT))
        self.typed(frame, text, ANNOT_PX, ANNOT_TRACK, W / 2, ANNOT_BASELINE, "c", GOLD,
                   op * ANNOT_ALPHA * (1 - ex), tin)

    def card(self, frame, kind, text, op, tin, tout):
        px, cy = CARD_STYLE[kind]
        g = self.glyphs(text, px, cy=cy)
        n = len(g)
        s = self.s
        stag = min(0.05, 0.75 / max(n - 1, 1))
        ex = 1 - min(1.0, max(0.0, tout / CARD_EXIT))
        # settles from wide tracking, then keeps opening very slowly while it holds
        track = 1 + CARD_TRACK * (1 - ease_out_expo(tin / CARD_SETTLE)) + 0.022 * min(1.0, tin / 5.0) + 0.05 * ex
        cxr = self.W / 2
        for k, (m, x0, y0, xc) in enumerate(g):
            p = ease_out((tin - k * stag) / CARD_FADE)
            o = op * p * (1 - ex) ** 1.2
            if o <= 1e-3:
                continue
            dx = int(round((xc - cxr) * (track - 1)))
            r = s * (3.5 * (1 - p) + 3.0 * ex)
            if r >= 0.3:
                mm, pad = self.blurred(m, r)
                blend(frame, mm, x0 + dx - pad, y0 - pad, CAPTION_RGB, o)
            else:
                blend(frame, m, x0 + dx, y0, CAPTION_RGB, o)

    def chapter(self, frame, texts, op, tin, tout):
        serif, mono = texts
        s = self.s
        ex = 1 - min(1.0, max(0.0, tout / CHAPTER_EXIT))
        o = op * CHAPTER_ALPHA * (1 - ex)
        if o <= 1e-3:
            return
        slide = int(round(-10 * s * ex))
        # hairline: draws on left to right
        ml, _, _, _, _ = self.mono_line(mono, CHAPTER_MONO_PX, 0.3, CHAPTER_X, CHAPTER_MONO_BASE, "l")
        g = self.glyphs(serif, CHAPTER_PX, cy=CHAPTER_BASE - CHAPTER_PX * 0.35, left=CHAPTER_X)
        x_end = max(g[-1][3] + CHAPTER_PX * s * 0.6, CHAPTER_X * s + ml.shape[1])
        L = (x_end - CHAPTER_X * s + 18 * s) * ease_out(tin / 0.45)
        if L > 0.5:
            th = max(1, int(round(1.0 * s)))
            line = np.full((th, int(math.ceil(L))), 255, np.uint8)
            frac = L - math.floor(L)
            if frac > 0:
                line[:, -1] = int(255 * frac)
            blend(frame, line, int(round(CHAPTER_X * s)) + slide, int(round(CHAPTER_LINE_Y * s)), GOLD, o * 0.55)
        stag = 0.045
        for k, (m, x0, y0, _) in enumerate(g):
            p = ease_out((tin - 0.08 - k * stag) / 0.32)
            if p <= 0:
                break
            blend(frame, m, x0 + slide, y0 + int(round(6 * s * (1 - p))), CAPTION_RGB, o * p)
        tm = tin - 0.3
        if tm > 0:
            self.typed(frame, mono, CHAPTER_MONO_PX, 0.3, CHAPTER_X + slide / s, CHAPTER_MONO_BASE, "l",
                       GOLD, o * 0.75, tm, cps=40.0, cursor=False)

    def hud(self, frame, text, op, tin, tout, params):
        ex = 1 - min(1.0, max(0.0, tout / 0.4))
        o = op * HUD_ALPHA * (1 - ex) * (1 + 0.6 * params.get("flash", 0.0))
        if o <= 1e-3:
            return
        self.typed(frame, text, HUD_PX, 0.12, HUD_RIGHT, HUD_BASE, "r", GOLD, o, tin, cps=70.0,
                   cursor=False)
        sp = params.get("spark")
        if sp and tin > 0.25:
            self.sparkline(frame, sp, o * min(1.0, (tin - 0.25) / 0.3))

    def sparkline(self, frame, sp, o):
        """sp: dict(vals, n_max, k, p, scale, ghost=(vals, n_max) | None). Draws values 0..k-1
        fully and the segment toward k at progress p, a dot at the head."""
        s = self.s
        x1 = HUD_RIGHT * s
        x0 = x1 - SPARK_W * s
        y0 = SPARK_TOP * s
        hh = SPARK_H * s
        pad = 4
        bx0, by0 = int(math.floor(x0)) - pad, int(math.floor(y0)) - pad
        bw, bh = int(math.ceil(SPARK_W * s)) + 2 * pad, int(math.ceil(hh)) + 2 * pad
        scale = sp.get("scale", 1.0)

        def pts(vals, n_max):
            return [(x0 + i / n_max * SPARK_W * s - bx0, y0 + hh * (1 - v / scale) - by0) for i, v in enumerate(vals)]

        def poly(P, upto, p, width):
            cov = np.zeros((bh, bw), np.float32)
            segs = [(P[i], P[i + 1], 1.0) for i in range(min(upto, len(P) - 1))]
            if upto < len(P) - 1 and p > 0:
                a, b = P[upto], P[upto + 1]
                segs.append((a, (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p), 1.0))
            for (ax, ay), (cx_, cy_), _ in segs:
                c = _seg_cov(ax, ay, cx_, cy_, width, bw, bh)
                if c is not None:
                    i0, i1, j0, j1, cv = c
                    np.maximum(cov[j0:j1, i0:i1], cv, out=cov[j0:j1, i0:i1])
            return cov
        # baseline (spread = 0)
        base = np.zeros((bh, bw), np.float32)
        yb = int(round(y0 + hh - by0))
        if 0 <= yb < bh:
            base[yb, pad:bw - pad] = 0.5
        blend(frame, (base * 255).astype(np.uint8), bx0, by0, GOLD, o * 0.35)
        if sp.get("ghost"):
            gv, gn = sp["ghost"]
            cov = poly(pts(gv, gn), len(gv) - 1, 0.0, max(1.0 * s, 0.6))
            blend(frame, (cov * 255).astype(np.uint8), bx0, by0, GOLD, o * 0.35)
        P = pts(sp["vals"], sp["n_max"])
        k, p = sp["k"], sp["p"]
        cov = poly(P, k, p, max(1.3 * s, 0.7))
        if k < len(P) - 1 and p > 0:
            hx = P[k][0] + (P[k + 1][0] - P[k][0]) * p
            hy = P[k][1] + (P[k + 1][1] - P[k][1]) * p
        else:
            hx, hy = P[min(k, len(P) - 1)]
        c = _seg_cov(hx, hy, hx, hy, max(4.2 * s, 1.6), bw, bh)
        if c is not None:
            i0, i1, j0, j1, cv = c
            np.maximum(cov[j0:j1, i0:i1], cv, out=cov[j0:j1, i0:i1])
        blend(frame, (np.clip(cov, 0, 1) * 255).astype(np.uint8), bx0, by0, GOLD, min(1.0, o * 1.6))

    def mono_at(self, frame, text, op, params):
        mask, x0, y0, _, _ = self.mono_line(text, params.get("px", 12), 0.1, params["x"], params["y"],
                                            params.get("align", "c"))
        blend(frame, mask, x0, y0, GOLD, op)

    # -- dispatch ----------------------------------------------------------------------
    def draw(self, frame, texts):
        for kind, text, op, prm in texts:
            if op <= 1e-3:
                continue
            tin, tout = prm.get("tin", 99.0), prm.get("tout", 99.0)
            if kind == "caption":
                self.caption(frame, text, op, tin, tout)
            elif kind == "annotation":
                self.annotation(frame, text, op, tin, tout)
            elif kind.startswith("card_"):
                self.card(frame, kind, text, op, tin, tout)
            elif kind == "chapter":
                self.chapter(frame, text, op, tin, tout)
            elif kind == "hud":
                self.hud(frame, text, op, tin, tout, prm)
            elif kind == "mono_at":
                self.mono_at(frame, text, op, prm)
            else:
                raise ValueError(kind)
