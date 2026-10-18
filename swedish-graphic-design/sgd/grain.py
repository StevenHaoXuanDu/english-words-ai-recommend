"""Static paper grain.

The grain is generated once from a fixed seed and applied identically to
every frame, so the texture reads as printed paper and never flickers.
"""

import numpy as np

from .core import W, H

_STATE = None


def _blur(img, passes):
    """Cheap separable box blur (repeated passes approximate a gaussian)."""
    for _ in range(passes):
        img = (np.roll(img, 1, 0) + img + np.roll(img, -1, 0)) / 3.0
        img = (np.roll(img, 1, 1) + img + np.roll(img, -1, 1)) / 3.0
    return img


def _upsample(small, h, w):
    yy = np.linspace(0, small.shape[0] - 1.001, h, dtype=np.float32)
    xx = np.linspace(0, small.shape[1] - 1.001, w, dtype=np.float32)
    y0 = yy.astype(int)
    x0 = xx.astype(int)
    fy = (yy - y0)[:, None]
    fx = (xx - x0)[None, :]
    return (small[y0][:, x0] * (1 - fy) * (1 - fx) + small[y0 + 1][:, x0] * fy * (1 - fx)
            + small[y0][:, x0 + 1] * (1 - fy) * fx + small[y0 + 1][:, x0 + 1] * fy * fx)


def _make():
    rng = np.random.default_rng(1969)
    fine = _blur(rng.standard_normal((H, W)).astype(np.float32), 1)
    fine /= fine.std()
    mott = _upsample(_blur(rng.standard_normal((H // 20 + 2, W // 20 + 2)).astype(np.float32), 2), H, W)
    mott /= mott.std()
    g = 0.0105 * fine + 0.0055 * mott
    # Fixed point multiplier: out = (in * mul) >> 7, mul around 128. All four
    # channels are processed so the arithmetic runs on contiguous memory.
    mul = np.clip(np.round(128.0 * (1.0 + g)), 0, 255).astype(np.uint16)
    mul4 = np.ascontiguousarray(np.repeat(mul[:, :, None], 4, axis=2))
    tmp = np.empty((H, W, 4), np.uint16)
    return mul4, tmp


def apply(buf):
    """Apply grain in place to a BGRx uint8 frame given as an (H, W, 4) view."""
    global _STATE
    if _STATE is None:
        _STATE = _make()
    mul4, tmp = _STATE
    np.multiply(buf, mul4, out=tmp, dtype=np.uint16, casting="unsafe")
    np.right_shift(tmp, 7, out=tmp)
    np.minimum(tmp, 255, out=tmp)
    np.copyto(buf, tmp, casting="unsafe")
