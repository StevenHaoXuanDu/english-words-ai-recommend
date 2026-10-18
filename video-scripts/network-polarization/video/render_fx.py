"""Ambient background and light effects for render.py, drawn behind the data layer.

Background(scale).draw(scene, t) returns a uint8 frame that render.py then composites the
data rects onto (Raster.compose). It holds, back to front:

* nebula   - a few very soft warm light clouds drifting slowly (a low-res cloud texture
             sampled at moving offsets, inside drifting gaussian envelopes), plus a faint
             haze in the colour of whatever data is on screen (light spill), vignetted.
             Computed on a coarse grid, upsampled, then dithered (interleaved-gradient
             noise, a fixed pattern: no temporal grain for the encoder) so the slow
             gradients never band.
* dust     - gold particles in three parallax depths drifting upward, twinkling; they
             follow the shot camera at their depth's parallax, smeared along their own
             screen motion so they never strobe during fast camera moves.
* axis     - the opinion axis 0 ... 1 under the row: a hairline with ticks every 0.1,
             transformed by the very camera the rects use (its ticks line up with x),
             drawing on in the opening, breathing, brightening on every round.
* effects  - scene.fx: ripples, light beams, round scanlines, spark bursts, flashes.

Nothing here touches a data rect; everything is light behind / around it, and every
element stays far darker than the rects (peak values are noted at each constant).
"""
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from render_core import GOLD, H, W, X, Y0, hex_rgb

NEB_W, NEB_H = 160, 90               # nebula grid: 12 design px per cell
NEB_PEAK = 0.100                     # brightest nebula value (sRGB 0..1; rects are ~0.75-0.9)
SPILL_CAP, SPILL_GAIN, SPILL_SIGMA = 0.040, 0.22, 4.5   # data-coloured haze (cells)
AXIS_DY = 190.0                      # axis: design px below the row centreline at zoom 1
AXIS_Y_PARALLAX = 0.35               # ... and how much of the zoom its height follows
AXIS_ALPHA, TICK_ALPHA = 0.20, 0.26  # hairline / tick brightness (x gold)
RIPPLE_FLOOR = 790.0                 # accent ripples fade to nothing by this design y
AMBER = hex_rgb("#C98F4A")
DUST_RGB = np.array([hex_rgb("#F2D9A0"), hex_rgb("#E8C066"), hex_rgb("#D9A25A")])


def ign(w, h):
    """Interleaved gradient noise in [0, 1): a fixed, blue-ish dither pattern."""
    x = np.arange(w, dtype=np.float64)[None, :]
    y = np.arange(h, dtype=np.float64)[:, None]
    return np.mod(52.9829189 * np.mod(0.06711056 * x + 0.00583715 * y, 1.0), 1.0).astype(np.float32)


def _cloud_texture(n=128, seed=7):
    """Tileable smooth noise in 0..1 (band-limited white noise)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal((n, n)))
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    k = np.hypot(kx, ky)
    amp = np.exp(-(k / 0.035) ** 2) + 0.35 * np.exp(-(k / 0.09) ** 2)
    a = np.real(np.fft.ifft2(f * amp))
    a = (a - a.min()) / (a.max() - a.min())
    return a.astype(np.float32)


def smoothstep(e0, e1, v):
    u = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def layer_xy(wx, wy, cam, drift, par):
    """Screen position (design px) of points in a parallax layer: camera zoom/pan and the
    global drift both act at `par` of their full strength (par = 1: the data plane)."""
    z, cx = cam
    dz, dx, dy = drift
    zp = 1.0 + (z - 1.0) * par
    cxp = 960.0 + (cx - 960.0) * par
    dzp = 1.0 + (dz - 1.0) * par
    sx = ((wx - cxp) * zp) * dzp + 960.0 + dx * par
    sy = ((wy - Y0) * zp) * dzp + Y0 + dy * par
    return sx, sy


# ----------------------------------------------------------------------------- dust
class Dust:
    """Deterministic particle field. Layers: (count, parallax, size px, alpha, speed px/s)."""
    LAYERS = [(140, 0.22, (0.65, 0.95), (0.18, 0.36), (4.0, 9.0)),
              (74, 0.55, (0.95, 1.45), (0.31, 0.59), (8.0, 15.0)),
              (26, 1.25, (2.2, 3.8), (0.14, 0.22), (15.0, 25.0))]
    MARGIN = 60.0

    def __init__(self, seed=1017):
        rng = np.random.default_rng(seed)
        cols = []
        for li, (n, par, sz, al, sp) in enumerate(self.LAYERS):
            d = dict(par=par,
                     x0=rng.uniform(0, W + 2 * self.MARGIN, n), y0=rng.uniform(0, H + 2 * self.MARGIN, n),
                     size=rng.uniform(*sz, n), alpha=rng.uniform(*al, n),
                     vy=-rng.uniform(*sp, n), vx=rng.uniform(-0.35, 0.35, n) * sp[1],
                     wob=rng.uniform(4, 14, n), wper=rng.uniform(7, 16, n), wph=rng.uniform(0, 2 * np.pi, n),
                     tw=rng.uniform(0.15, 0.45, n), tph=rng.uniform(0, 2 * np.pi, n),
                     rgb=DUST_RGB[rng.integers(0, 3, n)])
            cols.append(d)
        self.layers = cols

    def positions(self, t, cam, drift):
        """Per layer: (sx, sy) design px (screen space, wrapped), plus the layer dict."""
        out = []
        P, Q = W + 2 * self.MARGIN, H + 2 * self.MARGIN
        for L in self.layers:
            wx = L["x0"] + L["vx"] * t + L["wob"] * np.sin(2 * np.pi * t / L["wper"] + L["wph"])
            wy = L["y0"] + L["vy"] * t + 0.5 * L["wob"] * np.cos(2 * np.pi * t / (1.3 * L["wper"]) + L["wph"])
            sx, sy = layer_xy(wx - self.MARGIN, wy - self.MARGIN, cam, drift, L["par"])
            # wrapping in screen space keeps the density constant under any zoom
            sx = np.mod(sx + self.MARGIN, P) - self.MARGIN
            sy = np.mod(sy + self.MARGIN, Q) - self.MARGIN
            out.append((sx, sy, L))
        return out


def splat(frame, s, x, y, sigma, rgb, amp, vx=0.0, vy=0.0):
    """Add a gaussian light dot (design px centre / sigma, peak `amp`) to a uint8 frame,
    smeared along (vx, vy) design px (energy kept: longer smear, dimmer)."""
    if amp <= 1.5e-3:
        return
    x, y, sg = x * s, y * s, max(sigma * s, 0.45)
    vx, vy = vx * s, vy * s
    L = math.hypot(vx, vy)
    pad = 3.0 * sg + 1.0
    x0 = int(math.floor(min(x - vx / 2, x + vx / 2) - pad))
    x1 = int(math.ceil(max(x - vx / 2, x + vx / 2) + pad))
    y0 = int(math.floor(min(y - vy / 2, y + vy / 2) - pad))
    y1 = int(math.ceil(max(y - vy / 2, y + vy / 2) + pad))
    Hh, Ww = frame.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= Ww or y0 >= Hh:
        return
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, Ww), min(y1, Hh)
    px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - x
    py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - y
    if L > 0.5:
        ux, uy = vx / L, vy / L
        along = px * ux + py * uy
        across = -px * uy + py * ux
        a = np.clip(np.abs(along) - L / 2, 0.0, None)
        d2 = a * a + across * across
        amp = amp / (1.0 + L / (2.5 * sg))
    else:
        d2 = px * px + py * py
    k = np.exp(-d2 / (2 * sg * sg)) * (amp * 255.0)
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    reg += k[..., None] * np.asarray(rgb, np.float32)
    np.minimum(reg, 255.0, out=reg)
    frame[y0:y1, x0:x1] = reg.astype(np.uint8)


def add_region(frame, x0, y0, img):
    """Add a float RGB (0..1) image to the uint8 frame at integer (x0, y0), clipped."""
    h, w = img.shape[:2]
    Hh, Ww = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, Ww), min(y0 + h, Hh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    reg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    reg += sub * 255.0
    np.minimum(reg, 255.0, out=reg)
    frame[fy0:fy1, fx0:fx1] = (reg + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- background
class Background:
    def __init__(self, scale=1.0):
        self.s = scale
        self.W = int(round(W * scale / 2)) * 2
        self.H = int(round(H * scale / 2)) * 2
        self.hw, self.hh = self.W // 2, self.H // 2
        self.dither = ign(self.hw, self.hh)
        gx = (np.arange(NEB_W) + 0.5) / NEB_W * W
        gy = (np.arange(NEB_H) + 0.5) / NEB_H * H
        self.gx, self.gy = np.meshgrid(gx, gy)
        r = np.hypot((self.gx - 960) / 960, (self.gy - 540) / 600)
        self.vig = (1.0 - 0.72 * smoothstep(0.35, 1.25, r)).astype(np.float32)
        self.tex = _cloud_texture()
        self.dust = Dust()
        rng = np.random.default_rng(2024)
        # nebula envelopes: base centre, drift amplitude/period, sizes, weight
        self.blobs = [dict(c=(rng.uniform(300, 1620), rng.uniform(260, 820)),
                           a=(rng.uniform(150, 280), rng.uniform(70, 150)),
                           p=(rng.uniform(22, 40), rng.uniform(18, 34)), ph=rng.uniform(0, 6.3, 2),
                           sg=(rng.uniform(380, 620), rng.uniform(220, 340)), w=rng.uniform(0.55, 1.0))
                      for _ in range(5)]

    # -- nebula -------------------------------------------------------------------------
    def nebula(self, t, cam, drift, spill_src, level):
        gx, gy = self.gx, self.gy
        # the clouds sit far back: they follow the camera at 10 %
        z, cx = cam
        zp = 1.0 + (z - 1.0) * 0.10
        cxp = 960.0 + (cx - 960.0) * 0.10
        wx = (gx - 960.0) / zp + cxp
        wy = (gy - Y0) / zp + Y0
        env = np.zeros_like(gx)
        for b in self.blobs:
            bx = b["c"][0] + b["a"][0] * math.sin(2 * math.pi * t / b["p"][0] + b["ph"][0])
            by = b["c"][1] + b["a"][1] * math.sin(2 * math.pi * t / b["p"][1] + b["ph"][1])
            env += b["w"] * np.exp(-0.5 * (((wx - bx) / b["sg"][0]) ** 2 + ((wy - by) / b["sg"][1]) ** 2))
        # a soft horizon band of light behind the row
        env += 0.45 * np.exp(-0.5 * ((wy - Y0) / 150.0) ** 2) * (0.85 + 0.15 * math.sin(2 * math.pi * t / 9.0))
        env *= 0.8
        n = self.tex.shape[0]
        u1 = (wx / 1920.0 * 0.75 + t * 0.0080) * n
        v1 = (wy / 1080.0 * 0.45 + t * 0.0030) * n
        u2 = (wx / 1920.0 * 1.6 - t * 0.0110) * n
        v2 = (wy / 1080.0 * 0.95 + t * 0.0060) * n
        c1 = ndimage.map_coordinates(self.tex, [v1.ravel(), u1.ravel()], order=1, mode="grid-wrap").reshape(gx.shape)
        c2 = ndimage.map_coordinates(self.tex, [v2.ravel(), u2.ravel()], order=1, mode="grid-wrap").reshape(gx.shape)
        cloud = smoothstep(0.12, 0.92, 0.7 * c1 + 0.3 * c2)
        val = env * (0.40 + 0.60 * cloud)
        val = NEB_PEAK * (1.0 - np.exp(-val / 0.9)) * self.vig * level   # soft-saturating
        rgb = val[..., None] * AMBER[None, None, :].astype(np.float32) / AMBER.max()
        if spill_src is not None:
            sx, sy, col = spill_src
            if len(sx):
                acc = np.zeros((NEB_H, NEB_W, 3), np.float32)
                ix = np.clip(np.floor(sx / W * NEB_W).astype(int), 0, NEB_W - 1)
                iy = np.clip(np.floor(sy / H * NEB_H).astype(int), 0, NEB_H - 1)
                for ch in range(3):
                    np.add.at(acc[:, :, ch], (iy, ix), col[:, ch])
                acc = ndimage.gaussian_filter(acc, (SPILL_SIGMA, SPILL_SIGMA, 0), mode="constant")
                acc *= SPILL_GAIN
                m = acc.max(axis=2, keepdims=True)
                acc *= np.where(m > 1e-6, SPILL_CAP * (1 - np.exp(-m / SPILL_CAP)) / np.maximum(m, 1e-6), 0.0)
                rgb += acc * (0.6 + 0.4 * self.vig[..., None]) * level
        return rgb.astype(np.float32)

    # -- one frame ----------------------------------------------------------------------
    def draw(self, sc, t, fps=30.0, cam_prev=None, cam_next=None):
        s = self.s
        level = float(getattr(sc, "bg_level", 1.0)) * float(getattr(sc, "fade", 1.0))
        cam = getattr(sc, "cam", (1.0, 960.0))
        drift = getattr(sc, "drift", (1.0, 0.0, 0.0))
        spill = getattr(sc, "spill", None)
        lo = self.nebula(t, cam, drift, spill, level)
        half = np.empty((self.hh, self.hw, 3), np.uint8)
        for ch in range(3):
            a = np.asarray(Image.fromarray(np.ascontiguousarray(lo[:, :, ch]), mode="F")
                           .resize((self.hw, self.hh), Image.BILINEAR)).copy()
            a *= 255.0
            a += self.dither
            half[:, :, ch] = a
        frame = half.repeat(2, 0).repeat(2, 1)
        if level > 1e-3:
            self.draw_dust(frame, sc, t, fps, cam, drift, cam_prev, cam_next, level)
        if getattr(sc, "axis", None):
            self.draw_axis(frame, sc.axis, t)
        for fx in getattr(sc, "fx", []):
            FX[fx["type"]](frame, s, fx)
        return frame

    def draw_dust(self, frame, sc, t, fps, cam, drift, cam_prev, cam_next, level):
        s = self.s
        h = 0.5 / fps
        cur = self.dust.positions(t, cam, drift)
        if cam_prev is not None and cam_next is not None:
            a = self.dust.positions(t - h, cam_prev[0], cam_prev[1])
            b = self.dust.positions(t + h, cam_next[0], cam_next[1])
        else:
            a = b = None
        for li, (sx, sy, L) in enumerate(cur):
            tw = 1.0 - L["tw"] + L["tw"] * np.sin(2 * np.pi * t * 0.35 * (1 + li * 0.3) + L["tph"])
            vg = 1.0 - 0.75 * smoothstep(0.4, 1.2, np.hypot((sx - 960) / 960, (sy - 540) / 600))
            amp = L["alpha"] * tw * vg * level
            if a is not None:
                vx = b[li][0] - a[li][0]
                vy = b[li][1] - a[li][1]
                wrap = (np.abs(vx) > 200) | (np.abs(vy) > 200)        # crossed the wrap seam
                vx = np.where(wrap, 0.0, vx)
                vy = np.where(wrap, 0.0, vy)
            else:
                vx = vy = np.zeros_like(sx)
            vis = (sx > -20) & (sx < W + 20) & (sy > -20) & (sy < H + 20) & (amp > 0.004)
            for i in np.flatnonzero(vis):
                splat(frame, s, float(sx[i]), float(sy[i]), float(L["size"][i]), L["rgb"][i], float(amp[i]),
                      float(vx[i]), float(vy[i]))

    def draw_axis(self, frame, ax, t):
        """ax: dict(parts=[(cam, x_from, x_to, y)], drift, draw (0..1 from the centre out), alpha,
        pulse, glint). Each part is the axis under one camera, clipped to [x_from, x_to) screen
        design px, at screen y (the renderer keeps it clear of the rects)."""
        s = self.s
        drift = ax["drift"]
        a = ax["alpha"] * (0.85 + 0.15 * math.sin(2 * math.pi * t / 5.0)) * (1 + 0.8 * ax.get("pulse", 0.0))
        if a <= 2e-3 or ax["draw"] <= 1e-3:
            return
        dz, dx, dy = drift
        Hh, Ww = frame.shape[:2]
        for cam, c0, c1, ys in ax["parts"]:
            z = cam[0]
            xs = lambda v: (((float(X(v)) - cam[1]) * z) * dz + 960.0 + dx)
            xc = xs(0.5)
            half = (xs(1.0) - xs(0.0)) / 2 * ax["draw"]
            ea, eb = xc - half, xc + half                    # true ends (soft)
            xa, xb = max(ea, -10.0, c0), min(eb, W + 10.0, c1)
            if xb <= xa:
                continue
            # hairline (1 design px, at least 0.7 raster px), with a slow glint travelling along it
            y = ys * s
            th = max(1.0 * s, 0.7)
            i0, i1 = int(max(math.floor(xa * s), 0)), int(min(math.ceil(xb * s), Ww))
            j0, j1 = int(math.floor(y - th / 2)), int(math.floor(y + th / 2)) + 1
            if i1 > i0 and 0 <= j0 and j1 <= Hh:
                px = np.arange(i0, i1, dtype=np.float32) + 0.5
                g = ax.get("glint", None)
                prof = np.ones_like(px)
                if g is not None:
                    prof += 1.4 * np.exp(-0.5 * ((px / s - g) / 90.0) ** 2)
                prof *= np.clip((px - ea * s) / (30 * s), 0, 1) * np.clip((eb * s - px) / (30 * s), 0, 1)
                for j in range(j0, j1):
                    cov = min(j + 1, y + th / 2) - max(j, y - th / 2)
                    if cov <= 0:
                        continue
                    reg = frame[j, i0:i1].astype(np.float32)
                    reg += (a * AXIS_ALPHA * cov * 255.0) * prof[:, None] * GOLD.astype(np.float32)
                    frame[j, i0:i1] = np.minimum(reg, 255.0).astype(np.uint8)
            # ticks at 0, 0.1, ... 1 (longer at 0, 0.5, 1), appearing as the line reaches them
            for k in range(11):
                v = k / 10
                xt = xs(v)
                if xt < xa - 0.5 or xt > xb + 0.5 or not (c0 <= xt < c1):
                    continue
                ln = (12.0 if k in (0, 5, 10) else 6.0) * min(z, 1.6) ** 0.5
                tw = max(1.0 * s, 0.7)
                x = xt * s
                ii0, ii1 = int(math.floor(x - tw / 2)), int(math.floor(x + tw / 2)) + 1
                jj0, jj1 = int(math.floor(y)), int(math.ceil(y + ln * s))
                if ii0 < 0 or ii1 > Ww or jj0 < 0 or jj1 > Hh:
                    continue
                for i in range(ii0, ii1):
                    cov = min(i + 1, x + tw / 2) - max(i, x - tw / 2)
                    if cov <= 0:
                        continue
                    col = frame[jj0:jj1, i].astype(np.float32)
                    fall = np.linspace(1.0, 0.35, jj1 - jj0, dtype=np.float32)[:, None]
                    col += (a * TICK_ALPHA * cov * 255.0) * fall * GOLD.astype(np.float32)
                    frame[jj0:jj1, i] = np.minimum(col, 255.0).astype(np.uint8)


# ----------------------------------------------------------------------------- effects
def fx_ripple(frame, s, f):
    """Flat elliptical shock ring (a ripple across the 'floor' of the row)."""
    cx, cy, r, ratio, amp, wd = f["x"] * s, f["y"] * s, f["r"] * s, f.get("ratio", 0.28), f["amp"], f.get("w", 2.0)
    if amp <= 2e-3 or r < 2:
        return
    rx, ry = r, max(r * ratio, 1.0)
    sig = max(wd * s, 0.8)
    halo = 12.0 * s
    pad = 3 * halo + 2
    Hh, Ww = frame.shape[:2]
    x0, x1 = int(max(cx - rx - pad, 0)), int(min(cx + rx + pad, Ww))
    y0, y1 = int(max(cy - ry - pad, 0)), int(min(cy + ry + pad, Hh))
    if x1 <= x0 or y1 <= y0:
        return
    px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - cx
    py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cy
    q = np.sqrt((px / rx) ** 2 + (py / ry) ** 2) + 1e-6
    gx, gy = px / (rx * rx) / q, py / (ry * ry) / q
    d = (q - 1.0) / np.maximum(np.hypot(gx, gy), 1e-6)
    prof = 0.5 * np.exp(-0.5 * (d / sig) ** 2) + 0.5 * np.exp(-0.5 * (d / halo) ** 2)
    # fainter on the far (upper) side of the ellipse: reads as lying flat
    prof *= 0.55 + 0.45 * np.clip(0.5 + py / (2 * ry), 0, 1)
    # ... and gone before the caption band (design y 800-900): it never crosses the text
    yy = (py + cy) / s
    prof *= np.clip((RIPPLE_FLOOR - yy) / 70.0, 0.0, 1.0) ** 2
    img = (amp * prof)[..., None] * np.asarray(f.get("rgb", GOLD), np.float32)
    add_region(frame, x0, y0, img)


def fx_beam(frame, s, f):
    """Vertical light beam at screen x (design px): bright core + wide soft halo, strongest
    on the row, fading toward the top and bottom of the frame."""
    x, amp = f["x"] * s, f["amp"]
    if amp <= 2e-3:
        return
    Hh, Ww = frame.shape[:2]
    core, halo = 2.0 * s, 70.0 * s
    x0, x1 = int(max(x - 3 * halo, 0)), int(min(x + 3 * halo, Ww))
    if x1 <= x0:
        return
    px = np.arange(x0, x1, dtype=np.float32) + 0.5 - x
    prof_x = 0.55 * np.exp(-0.5 * (px / core) ** 2) + 0.45 * np.exp(-0.5 * (px / halo) ** 2)
    py = np.arange(Hh, dtype=np.float32) + 0.5 - Y0 * s
    prof_y = np.exp(-0.5 * (py / (300.0 * s)) ** 2)
    img = (amp * prof_y[:, None] * prof_x[None, :])[..., None] * np.asarray(f.get("rgb", (1.0, 0.93, 0.78)), np.float32)
    add_region(frame, x0, 0, img)


def fx_scan(frame, s, f):
    """Horizontal scanline of light across the data's x extent at screen y."""
    y, xa, xb, amp = f["y"] * s, f["x0"] * s, f["x1"] * s, f["amp"]
    if amp <= 2e-3 or xb <= xa:
        return
    Hh, Ww = frame.shape[:2]
    core, halo, ext = 1.2 * s, 9.0 * s, 24.0 * s
    y0, y1 = int(max(y - 3 * halo, 0)), int(min(y + 3 * halo, Hh))
    x0, x1 = int(max(xa - ext, 0)), int(min(xb + ext, Ww))
    if y1 <= y0 or x1 <= x0:
        return
    py = np.arange(y0, y1, dtype=np.float32) + 0.5 - y
    prof_y = 0.6 * np.exp(-0.5 * (py / core) ** 2) + 0.4 * np.exp(-0.5 * (py / halo) ** 2)
    px = np.arange(x0, x1, dtype=np.float32) + 0.5
    prof_x = np.clip((px - (xa - ext)) / (2 * ext), 0, 1) * np.clip(((xb + ext) - px) / (2 * ext), 0, 1)
    prof_x = prof_x * prof_x * (3 - 2 * prof_x)
    img = (amp * prof_y[:, None] * prof_x[None, :])[..., None] * GOLD.astype(np.float32)
    add_region(frame, x0, y0, img)


def fx_sparks(frame, s, f):
    """Spark burst: f['pts'] = list of (x, y, vx, vy, size, amp) in design px / px per frame."""
    rgb = np.asarray(f.get("rgb", GOLD), np.float32)
    for x, y, vx, vy, sz, amp in f["pts"]:
        splat(frame, s, x, y, sz, rgb, amp, vx, vy)


def fx_glow(frame, s, f):
    """Large soft radial glow (an accent flash), elliptical, centred at (x, y)."""
    cx, cy, amp = f["x"] * s, f["y"] * s, f["amp"]
    if amp <= 2e-3:
        return
    rx, ry = f["rx"] * s, f["ry"] * s
    Hh, Ww = frame.shape[:2]
    x0, x1 = int(max(cx - 2.5 * rx, 0)), int(min(cx + 2.5 * rx, Ww))
    y0, y1 = int(max(cy - 2.5 * ry, 0)), int(min(cy + 2.5 * ry, Hh))
    if x1 <= x0 or y1 <= y0:
        return
    px = (np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - cx) / rx
    py = (np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cy) / ry
    img = (amp * np.exp(-0.5 * (px * px + py * py)))[..., None] * np.asarray(f.get("rgb", GOLD), np.float32)
    add_region(frame, x0, y0, img)


def fx_pulses(frame, s, f):
    """Soft light pulses along a horizontal thread: f['y'], f['xs'] (centres), f['amp']."""
    rgb = np.asarray(f.get("rgb", GOLD), np.float32)
    for x, a in zip(f["xs"], f["amps"]):
        splat(frame, s, x, f["y"], f.get("sigma", 5.0), rgb, a, f.get("vx", 0.0), 0.0)


FX = {"ripple": fx_ripple, "beam": fx_beam, "scan": fx_scan, "sparks": fx_sparks,
      "glow": fx_glow, "pulses": fx_pulses}
