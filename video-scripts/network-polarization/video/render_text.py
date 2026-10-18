"""Animated typography for render.py, drawn last, straight onto the finished frame.

Scene.texts holds (kind, text, opacity, params). params carry the animation clock:
tin = seconds since the element appeared, tout = seconds until it disappears.

    caption      serif line; characters rise and fade in on a fast stagger, the line leaves
                 with a quick fade + blur. Fully readable from tin = REVEAL_MAX on.
    annotation   small mono line at the bottom; types on with a block cursor that blinks a
                 few times once the line is complete, fades out.
    card_s04/22  serif title card; per-character focus-in while the tracking settles, then the
                 tracking keeps opening very slowly. Glyphs sit at sub-pixel positions (4x
                 supersampled phases, interpolated), so the slow open glides instead of ticking.
    chapter      ("01 · 村子", "THE VILLAGE") top-left label with a drawn hairline.
    hud          top-right mono readout + optional sparkline (params["spark"]).
    mono_at      small mono text at a design-px position (axis labels).

Text glow: every element carries a warm gold halo in the language of the rect glow - a tight
and a wide gaussian of its own glyph mask - screen-blended into the frame *under* the crisp
glyph (the glyph itself is never blurred by it, and the screen blend can never reach white).
The halo follows each glyph's reveal (it blooms in with the stagger, with a slight flare),
breathes very slowly while the text holds and fades with the exit. Serif text glows at
about 0.3-0.4 peak, the small mono annotations / HUD at about a third of that.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from render_core import (ANNOT_ALPHA, ANNOT_BASELINE, ANNOT_PX, ANNOT_TRACK, CAPTION_PX, CAPTION_RGB,
                         CAPTION_Y, CARD_STYLE, GOLD, SERIF, SERIF_INDEX, W, TextCache, _seg_cov)

# caption reveal: char k starts at k * stagger; the whole line is in by REVEAL_MAX
CAP_FADE, CAP_STAG_MAX, CAP_STAG_TOTAL, CAP_RISE = 0.16, 0.025, 0.16, 8.0
REVEAL_MAX = CAP_STAG_TOTAL + CAP_FADE
CAP_EXIT, CAP_EXIT_BLUR = 0.22, 3.5
TYPE_CPS, CURSOR_HOLD, CURSOR_PERIOD = 55.0, 0.95, 0.5
ANNOT_EXIT = 0.25
ANNOT_TYPE_MAX = 0.6                      # s
CARD_FADE, CARD_STAG_TOTAL, CARD_TRACK, CARD_SETTLE, CARD_EXIT = 0.42, 0.5, 0.16, 1.1, 0.45
CARD_OPEN, CARD_OPEN_T = 0.022, 5.0      # slow tracking open while the card holds
CARD_IN = CARD_STAG_TOTAL + CARD_FADE    # every glyph of a card is in by then
CARD_GLINT = ((0.5, 3.9), 1.6)           # halo glints: start these long after CARD_IN, last
CHAPTER_X, CHAPTER_BASE, CHAPTER_PX = 120.0, 104.0, 30
CHAPTER_MONO_PX, CHAPTER_MONO_BASE, CHAPTER_LINE_Y = 13, 146.0, 121.0
CHAPTER_ALPHA, CHAPTER_EXIT = 0.82, 0.4
HUD_RIGHT, HUD_BASE, HUD_PX, HUD_ALPHA = 1800.0, 92.0, 14, 0.42
SPARK_W, SPARK_H, SPARK_TOP = 150.0, 26.0, 108.0
SS = 4                                   # card glyph supersampling (sub-pixel phases)

# text glow: (tight sigma, wide sigma) design px, (tight gain, wide gain), peak amplitude
GLOW_CAPTION = (2.2, 10.0, 0.75, 1.25, 0.42)
GLOW_CARD = (2.8, 16.0, 0.70, 1.35, 0.46)
GLOW_CHAPTER = (2.0, 8.0, 0.75, 1.20, 0.40)
GLOW_MONO = (1.4, 5.0, 0.70, 1.20, 0.16)
GLOW_CAP = 0.45                          # amp * halo never exceeds this
GLOW_RGB = np.asarray(GOLD, np.float32)


def ease_out(p):
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


def ease_out_expo(p):
    p = min(1.0, max(0.0, p))
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def bump(p):
    return math.sin(math.pi * p) if 0.0 < p < 1.0 else 0.0


def flare(p):
    """Halo strength over a glyph's reveal progress p: blooms in with it, slightly over 1."""
    return p + 0.45 * bump(p)


def breathe(tin):
    """Very slow halo breathing while text holds (+-7 %)."""
    return 1.0 + 0.07 * math.sin(2 * math.pi * tin / 3.3)


def _clip(frame, mask, x0, y0):
    h, w = mask.shape
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return fx0, fy0, fx1, fy1, mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]


def blend(frame, mask, x0, y0, col, opacity):
    """Alpha-blend a uint8/float mask (0..255) of colour col into the frame at (x0, y0)."""
    if opacity <= 1e-3:
        return
    c = _clip(frame, mask, x0, y0)
    if c is None:
        return
    fx0, fy0, fx1, fy1, sub = c
    m = sub.astype(np.float32) * (opacity / 255.0)
    reg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    reg += (np.asarray(col, np.float32) * 255.0 - reg) * m[..., None]
    frame[fy0:fy1, fx0:fx1] = (reg + 0.5).astype(np.uint8)


def glow(frame, halo, x0, y0, amp, col=GLOW_RGB):
    """Screen-blend a halo (float, ~0..1) tinted col at strength amp: out = 1 - (1-bg)(1-k col).
    Never reaches white; k is capped at GLOW_CAP."""
    if amp <= 2e-3:
        return
    c = _clip(frame, halo, x0, y0)
    if c is None:
        return
    fx0, fy0, fx1, fy1, sub = c
    k = np.minimum(sub * amp, GLOW_CAP)
    reg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    reg = 255.0 - (255.0 - reg) * (1.0 - k[..., None] * col)
    frame[fy0:fy1, fx0:fx1] = (reg + 0.5).astype(np.uint8)


def shift_x(m, fx):
    """Float copy of mask m moved fx (0..1) px to the right by linear interpolation (one
    column wider)."""
    m = m.astype(np.float32)
    out = np.zeros((m.shape[0], m.shape[1] + 1), np.float32)
    out[:, :-1] += m * (1.0 - fx)
    out[:, 1:] += m * fx
    return out


def halo_of(mask, s, spec):
    """Halo of a uint8/float 0..255 mask: (float array ~0..1, pad)."""
    s1, s2, g1, g2, _ = spec
    pad = int(math.ceil(2.6 * s2 * s)) + 1
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = np.clip(m + 0.5, 0, 255).astype(np.uint8)
    im = Image.fromarray(np.pad(m, pad))
    a = np.asarray(im.filter(ImageFilter.GaussianBlur(max(s1 * s, 0.4))), np.float32)
    b = np.asarray(im.filter(ImageFilter.GaussianBlur(max(s2 * s, 0.6))), np.float32)
    return (g1 / 255.0) * a + (g2 / 255.0) * b, pad


class TextFX:
    def __init__(self, scale, w, h):
        self.s, self.W, self.H = scale, w, h
        self.base = TextCache(scale, w, h)
        self.cache = {}
        self._mono = {}
        self._f4 = {}

    def _get(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
            if len(self.cache) > 900:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def halo(self, key, mask, spec):
        """Cached halo of a mask (key must identify the mask)."""
        return self._get(("halo", key, spec[:4]), lambda: halo_of(mask, self.s, spec))

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

    def glyph_halos(self, gkey, g, spec):
        """Per glyph: (halo, x0, y0) of the glyph run g (cached under gkey)."""
        def build():
            out = []
            for k, (m, x0, y0, _) in enumerate(g):
                h, pad = halo_of(m, self.s, spec)
                out.append((h, x0 - pad, y0 - pad))
            return out
        return self._get(("ghalos", gkey, spec[:4]), build)

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

    def line_and_halo(self, gkey, g, spec):
        """(x0, y0, mask, halo, hx0, hy0) of a whole glyph run, cached."""
        def build():
            x0, y0, m = self.line_mask(g)
            h, pad = halo_of(m, self.s, spec)
            return x0, y0, m, h, x0 - pad, y0 - pad
        return self._get(("line", gkey, spec[:4]), build)

    def blurred(self, mask, r):
        if r < 0.3:
            return mask, 0
        pad = int(math.ceil(3 * r))
        m = np.asarray(mask)
        if m.dtype != np.uint8:
            m = np.clip(m + 0.5, 0, 255).astype(np.uint8)
        im = Image.fromarray(np.pad(m, pad))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(r))), pad

    # -- card glyphs: supersampled sub-pixel phases --------------------------------------
    def card_glyphs(self, kind, text):
        """Per glyph: (phases, x0, y0, centre x). phases[j] (uint8) is the glyph drawn j/SS px
        to the right of its nominal place, from a SS x supersampled rendering (unhinted, so
        every phase has the same sharpness)."""
        px, cy = CARD_STYLE[kind]

        def build():
            s = self.s
            size = int(round(px * s * SS))
            if size not in self._f4:
                self._f4[size] = ImageFont.truetype(SERIF, size, index=SERIF_INDEX)
            f = self._f4[size]
            total = f.getlength(text) / SS
            ay = cy * s
            xs = self.W / 2 - total / 2
            asc, desc = f.getmetrics()
            hgt = int((asc + desc) / SS + 8)
            pad = int(math.ceil(px * s * 0.25)) + 2
            out = []
            for k, ch in enumerate(text):
                if ch.isspace():
                    continue
                xk = xs + f.getlength(text[:k]) / SS
                adv = f.getlength(ch) / SS
                x0 = int(math.floor(xk)) - pad
                y0 = int(math.floor(ay - hgt / 2))
                w1 = int(math.ceil(adv)) + 2 * pad + 2
                im = Image.new("L", ((w1 + 1) * SS, hgt * SS), 0)
                ImageDraw.Draw(im).text(((xk - x0) * SS, (ay - y0) * SS), ch, font=f, fill=255, anchor="lm")
                big = np.asarray(im, np.float32)
                phases = []
                for j in range(SS):
                    sh = np.zeros_like(big)
                    sh[:, j:] = big[:, :big.shape[1] - j]
                    ds = sh.reshape(hgt, SS, w1 + 1, SS).mean(axis=(1, 3))
                    phases.append(np.clip(ds + 0.5, 0, 255).astype(np.uint8))
                out.append((phases, x0, y0, xk + adv / 2))
            return out
        return self._get(("cardg", kind, text), build)

    @staticmethod
    def phase_mask(phases, xf):
        """(mask, integer x offset) of a glyph moved xf raster px (float) from its place."""
        ix = int(math.floor(xf))
        ph = (xf - ix) * SS
        j0 = int(ph)
        w = ph - j0
        a = phases[j0]
        if w < 1e-3:
            return a, ix
        if j0 + 1 < SS:
            b = phases[j0 + 1]
        else:                                    # next phase = phase 0 one px to the right
            b = np.zeros_like(phases[0])
            b[:, 1:] = phases[0][:, :-1]
        return a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w, ix

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

    def typed(self, frame, text, px, track, x, base, align, col, op, tin, cps=TYPE_CPS, cursor=True,
              glow_amp=1.0):
        """Type-on of a mono line with a block cursor; both carry a faint halo."""
        mask, x0, y0, cell, xs = self.mono_line(text, px, track, x, base, align)
        n = len(text)
        k = n if cps <= 0 else min(n, int(tin * cps))
        if k <= 0 and not cursor:
            return
        ga = GLOW_MONO[4] * op * glow_amp
        if k > 0:
            cut = max(0, min(int(round(xs + k * cell - x0)), mask.shape[1]))
            sub = mask[:, :cut]
            if ga > 2e-3 and cut > 0:
                if k == n:
                    h, pad = self.halo(("mono", text, px, track, x, base, align), mask, GLOW_MONO)
                else:
                    h, pad = halo_of(sub, self.s, GLOW_MONO)
                glow(frame, h, x0 - pad, y0 - pad, ga)
            blend(frame, sub, x0, y0, col, op)
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
        bx, by = int(round(cx0)), b - ch + int(round(0.12 * px * s))
        h, pad = self.halo(("cursor", cw, ch), blk, GLOW_MONO)
        glow(frame, h, bx - pad, by - pad, ga * 0.85 * c)
        blend(frame, blk, bx, by, col, op * 0.85 * c)

    # -- elements ----------------------------------------------------------------------
    def caption(self, frame, text, op, tin, tout):
        gkey = ("cap", text)
        g = self.glyphs(text, CAPTION_PX, cy=CAPTION_Y)
        n = len(g)
        stag = min(CAP_STAG_MAX, CAP_STAG_TOTAL / max(n - 1, 1))
        ex = 1 - min(1.0, max(0.0, tout / CAP_EXIT))
        s = self.s
        amp = GLOW_CAPTION[4] * op * breathe(tin)
        if ex > 0:                                  # leaving: whole line, fade + blur + drift up
            x0, y0, m, h, hx, hy = self.line_and_halo(gkey, g, GLOW_CAPTION)
            dy = -int(round(4 * s * ex))
            o = (1 - ex) ** 1.3
            glow(frame, h, hx, hy + dy, amp * o)
            r = CAP_EXIT_BLUR * s * ex
            if r >= 0.3:
                m, pad = self.blurred(m, r)
                x0, y0 = x0 - pad, y0 - pad
            blend(frame, m, x0, y0 + dy, CAPTION_RGB, op * o)
            return
        if tin >= (n - 1) * stag + CAP_FADE:
            x0, y0, m, h, hx, hy = self.line_and_halo(gkey, g, GLOW_CAPTION)
            glow(frame, h, hx, hy, amp)
            blend(frame, m, x0, y0, CAPTION_RGB, op)
            return
        halos = self.glyph_halos(gkey, g, GLOW_CAPTION)
        items = []
        for k, (m, x0, y0, _) in enumerate(g):
            p = ease_out((tin - k * stag) / CAP_FADE)
            if p <= 0:
                break
            items.append((k, p, int(round(CAP_RISE * s * (1 - p)))))
        for k, p, dy in items:
            h, hx, hy = halos[k]
            glow(frame, h, hx, hy + dy, amp * flare(p))
        for k, p, dy in items:
            m, x0, y0, _ = g[k]
            blend(frame, m, x0, y0 + dy, CAPTION_RGB, op * p)

    def annotation(self, frame, text, op, tin, tout):
        ex = 1 - min(1.0, max(0.0, tout / ANNOT_EXIT))
        # long lines type faster: every annotation is complete within ANNOT_TYPE_MAX
        self.typed(frame, text, ANNOT_PX, ANNOT_TRACK, W / 2, ANNOT_BASELINE, "c", GOLD,
                   op * ANNOT_ALPHA * (1 - ex), tin, cps=max(TYPE_CPS, len(text) / ANNOT_TYPE_MAX))

    def card(self, frame, kind, text, op, tin, tout):
        g = self.card_glyphs(kind, text)
        halos = self.glyph_halos(("card", kind, text), [(ph[0], x0, y0, xc) for ph, x0, y0, xc in g], GLOW_CARD)
        n = len(g)
        s = self.s
        stag = min(0.05, CARD_STAG_TOTAL / max(n - 1, 1))
        ex = 1 - min(1.0, max(0.0, tout / CARD_EXIT))
        # settles from wide tracking, then keeps opening very slowly while it holds
        track = (1 + CARD_TRACK * (1 - ease_out_expo(tin / CARD_SETTLE))
                 + CARD_OPEN * math.sin(0.5 * math.pi * min(1.0, tin / CARD_OPEN_T)) + 0.05 * ex)
        cxr = self.W / 2
        amp = GLOW_CARD[4] * op * breathe(tin) * (1 - ex) ** 1.2
        items = []
        for k, (phases, x0, y0, xc) in enumerate(g):
            p = ease_out((tin - k * stag) / CARD_FADE)
            o = op * p * (1 - ex) ** 1.2
            if o <= 1e-3:
                continue
            items.append((k, p, o, (xc - cxr) * (track - 1)))
        # once the card is in, a soft glint of light sweeps through its halo, left to right (again
        # later if the card holds that long)
        xg = None
        for t0 in CARD_GLINT[0]:
            u = (tin - CARD_IN - t0) / CARD_GLINT[1]
            if 0.0 < u < 1.0:
                break
        if 0.0 < u < 1.0:
            x_l, x_r = g[0][3] - 200 * s, g[-1][3] + 200 * s
            xg = x_l + (x_r - x_l) * (u * u * (3 - 2 * u))
        for k, p, o, dx in items:                    # halos first, then the glyphs over them
            h, hx, hy = halos[k]
            ix = int(math.floor(dx))
            gl = 1.0
            if xg is not None:
                gl += 0.9 * math.exp(-0.5 * ((g[k][3] + dx - xg) / (110 * s)) ** 2)
            glow(frame, shift_x(h, dx - ix), hx + ix, hy, amp * flare(p) * gl)
        for k, p, o, dx in items:
            phases, x0, y0, _ = g[k]
            m, ix = self.phase_mask(phases, dx)
            r = s * (3.5 * (1 - p) + 3.0 * ex)
            if r >= 0.3:
                mm, pad = self.blurred(m, r)
                blend(frame, mm, x0 + ix - pad, y0 - pad, CAPTION_RGB, o)
            else:
                blend(frame, m, x0 + ix, y0, CAPTION_RGB, o)

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
        gy = CHAPTER_BASE - CHAPTER_PX * 0.35
        g = self.glyphs(serif, CHAPTER_PX, cy=gy, left=CHAPTER_X)
        x_end = max(g[-1][3] + CHAPTER_PX * s * 0.6, CHAPTER_X * s + ml.shape[1])
        L = (x_end - CHAPTER_X * s + 18 * s) * ease_out(tin / 0.45)
        if L > 0.5:
            th = max(1, int(round(1.0 * s)))
            line = np.full((th, int(math.ceil(L))), 255, np.uint8)
            frac = L - math.floor(L)
            if frac > 0:
                line[:, -1] = int(255 * frac)
            lx, ly = int(round(CHAPTER_X * s)) + slide, int(round(CHAPTER_LINE_Y * s))
            h, pad = self.halo(("hair", line.shape), line, GLOW_MONO)
            glow(frame, h, lx - pad, ly - pad, GLOW_MONO[4] * 1.3 * o)
            blend(frame, line, lx, ly, GOLD, o * 0.55)
        stag = 0.045
        halos = self.glyph_halos(("chap", serif), g, GLOW_CHAPTER)
        amp = GLOW_CHAPTER[4] * o * breathe(tin)
        items = []
        for k, (m, x0, y0, _) in enumerate(g):
            p = ease_out((tin - 0.08 - k * stag) / 0.32)
            if p <= 0:
                break
            items.append((k, p, int(round(6 * s * (1 - p)))))
        for k, p, dy in items:
            h, hx, hy = halos[k]
            glow(frame, h, hx + slide, hy + dy, amp * flare(p))
        for k, p, dy in items:
            m, x0, y0, _ = g[k]
            blend(frame, m, x0 + slide, y0 + dy, CAPTION_RGB, o * p)
        tm = tin - 0.3
        if tm > 0:
            self.typed(frame, mono, CHAPTER_MONO_PX, 0.3, CHAPTER_X + slide / s, CHAPTER_MONO_BASE, "l",
                       GOLD, o * 0.75, tm, cps=40.0, cursor=False, glow_amp=1.3)

    def hud(self, frame, text, op, tin, tout, params):
        ex = 1 - min(1.0, max(0.0, tout / 0.4))
        o = op * HUD_ALPHA * (1 - ex) * (1 + 0.6 * params.get("flash", 0.0))
        if o <= 1e-3:
            return
        self.typed(frame, text, HUD_PX, 0.12, HUD_RIGHT, HUD_BASE, "r", GOLD, o, tin, cps=70.0,
                   cursor=False, glow_amp=1.4)
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
        m = (np.clip(cov, 0, 1) * 255).astype(np.uint8)
        h, hp = halo_of(m, s, GLOW_MONO)
        glow(frame, h, bx0 - hp, by0 - hp, GLOW_MONO[4] * 1.4 * min(1.0, o * 1.6))
        blend(frame, m, bx0, by0, GOLD, min(1.0, o * 1.6))

    def mono_at(self, frame, text, op, params):
        mask, x0, y0, _, _ = self.mono_line(text, params.get("px", 12), 0.1, params["x"], params["y"],
                                            params.get("align", "c"))
        h, pad = halo_of(mask, self.s, GLOW_MONO)
        glow(frame, h, x0 - pad, y0 - pad, GLOW_MONO[4] * op * params.get("glow", 1.0))
        blend(frame, mask, x0, y0, params.get("rgb", GOLD), op)

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
