"""Render stills and contact sheets for reviewing the film.

    python3 preview.py still 12.5 13 14          # full size PNGs
    python3 preview.py sheet 0 20 0.5            # contact sheet t0..t1 step dt
    python3 preview.py sheet 0 20 0.5 --cols 6
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from sgd.core import W, H
from sgd.film import Renderer

OUT = os.environ.get("PREVIEW_DIR", os.path.join(os.path.dirname(__file__), "build", "preview"))


def to_image(buf):
    return Image.frombuffer("RGBA", (W, H), bytes(buf), "raw", "BGRA", 0, 1).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "sheet"])
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=384)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    r = Renderer()
    if args.mode == "still":
        for t in args.times:
            img = to_image(r.render(t))
            path = os.path.join(OUT, f"still_{t:07.3f}.png")
            img.save(path)
            print(path)
        return
    t0, t1, dt = args.times
    times = []
    t = t0
    while t <= t1 + 1e-9:
        times.append(round(t, 4))
        t += dt
    tw = args.thumb
    th = tw * H // W
    cols = args.cols
    rows = (len(times) + cols - 1) // cols
    pad = 6
    label = 18
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + pad + label) + pad), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Medium.otf", 13)
    except OSError:
        font = None
    for i, t in enumerate(times):
        img = to_image(r.render(t)).resize((tw, th), Image.LANCZOS)
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + pad + label)
        sheet.paste(img, (x, y + label))
        draw.text((x, y + 1), f"{t:.2f}s  f{int(round(t * 30))}", fill=(220, 220, 220), font=font)
    name = args.name or f"sheet_{t0:06.2f}_{t1:06.2f}.png"
    path = os.path.join(OUT, name)
    sheet.save(path)
    print(path)


if __name__ == "__main__":
    main()
