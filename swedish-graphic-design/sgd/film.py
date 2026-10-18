"""Frame assembly: scene dispatch, final fade and paper grain."""

import cairo
import numpy as np

from . import grain
from .core import W, H, PAPER, set_color, prog, linear
from .scenes import (s01_form, s02_geometry, s03_grid, s04_color,
                     s05_type, s06_systems, s07_disruption, s08_final)

SCENES = [s01_form, s02_geometry, s03_grid, s04_color,
          s05_type, s06_systems, s07_disruption, s08_final]

# Final fade: the closing composition (still from s08_final.T_HOLD) fades
# cleanly to black and stays black to the last frame.
FADE_START = 178.0
FADE_END = 179.6


def draw(ctx, t):
    set_color(ctx, PAPER)
    ctx.paint()
    for scene in SCENES:
        if scene.START <= t < scene.END:
            ctx.save()
            scene.draw(ctx, t)
            ctx.restore()
            break
    a = prog(t, FADE_START, FADE_END - FADE_START, linear)
    if a > 0:
        ctx.set_source_rgba(0, 0, 0, a)
        ctx.paint()


def events():
    """Audio cue list gathered from every scene: (time, kind, params)."""
    out = []
    for scene in SCENES:
        out.extend(getattr(scene, "EVENTS", []))
    return sorted(out, key=lambda e: e[0])


class Renderer:
    def __init__(self, with_grain=True):
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        self.ctx = cairo.Context(self.surface)
        self.buf = np.ndarray((H, W, 4), np.uint8, self.surface.get_data())
        self.with_grain = with_grain

    def render(self, t):
        ctx = self.ctx
        ctx.save()
        ctx.reset_clip()
        ctx.new_path()
        draw(ctx, t)
        ctx.restore()
        self.surface.flush()
        if self.with_grain:
            grain.apply(self.buf)
        self.surface.mark_dirty()
        return self.buf
