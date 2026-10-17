"""Frame renderer for 「为什么网络让人变成二极管」 (no audio).

    python3 render.py --out BUILD/render/video_silent.mp4 [--workers 4]
    python3 render.py --stills 12.5,40.2 --png-dir DIR [--scale 0.5]
    python3 render.py --from T0 --to T1 --out X.mp4 [--scale 0.5] [--fps 15]
    python3 render.py --bench 12 [--scale 1]      # time single frames, no output
    add --no-mb to any of these to switch motion blur off (debugging only)

Reads timeline.json (--timeline to override) at run time; every opinion and
position comes from ../hk_sim.py (see render_shots.py).

Each frame: render_fx.Background (nebula, parallax dust, opinion axis, light effects)
-> the data layer (render_core.Raster, motion-blurred when anything moves) composited
over it in linear light -> render_text (animated captions, annotations, cards, chapter
labels, HUD). Global motion (drift, ambient life, round pulses, trails) is in
render_motion; per-shot choreography in render_shots.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BUILD = os.environ.get(
    "BUILD",
    "/tmp/claude-0/-home-user-english-words-ai-recommend/"
    "4af3582d-b5b8-57c7-bf82-51aca3b2d552/scratchpad/build",
)
DEFAULT_TIMELINE = os.path.join(BUILD, "timeline.json")
# aq-mode 3 biases bits toward dark flat areas: the dithered background haze keeps its
# smooth gradients instead of breaking into blocks (it still averages well under 60 MB)
X264_PARAMS = "aq-mode=3:aq-strength=0.9"

_show = None
_raster = None
_fps = 30.0
_mb = True


_bg = None
_text = None


def _init(timeline_path, scale, mb=True, fps=None):
    global _show, _raster, _fps, _mb, _bg, _text
    from render_core import Raster
    from render_fx import Background
    from render_shots import Show
    from render_text import TextFX
    with open(timeline_path) as f:
        _show = Show(json.load(f))
    _raster = Raster(scale)
    _bg = Background(scale)
    _text = TextFX(scale, _raster.W, _raster.H)
    _fps = float(fps or _show.fps)        # output frame rate sets the shutter interval
    _mb = mb


def render_at(t):
    """One output frame: background (render_fx), the data layer composited over it in linear
    light, then the typography (render_text). When rects or lines move, the data layer is
    motion-blurred over a 360 deg shutter (Show.motion_bounds + Raster.layers_blurred)."""
    sc = _show.scene(t)
    bounds = _show.motion_bounds(t, _fps, _raster.s) if _mb else None
    if bounds is None:
        layers = [_raster.layer(sc)]
    else:
        mids = [0.5 * (a + b) for a, b in zip(bounds[:-1], bounds[1:])]
        layers = _raster.layers_blurred([_show.scene(u, False) for u in mids],
                                        [_show.scene(u, False) for u in bounds])
    h = 0.5 / _fps
    cp, cn = _show.cam_at(t - h), _show.cam_at(t + h)
    frame = _bg.draw(sc, t, _fps, cp, cn)
    _raster.compose(frame, layers)
    _text.draw(frame, sc.texts)
    return frame


def _frame(t):
    return render_at(t).tobytes()


def _png(job):
    t, path = job
    from PIL import Image
    Image.fromarray(render_at(t)).save(path)
    return path


def _ordered(pool, fn, items, window):
    """Ordered results with bounded look-ahead (Pool.imap would buffer without limit)."""
    it = iter(items)
    pending = collections.deque()
    for x in it:
        pending.append(pool.apply_async(fn, (x,)))
        if len(pending) >= window:
            break
    while pending:
        r = pending.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.apply_async(fn, (nxt,)))
        yield r


def encode(args, fps, t0, t1):
    n = int(round((t1 - t0) * fps))
    times = [t0 + i / fps for i in range(n)]
    w, h = _raster.W, _raster.H
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rate = str(Fraction(fps).limit_denominator(1001))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", rate, "-i", "-", "-frames:v", str(n),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-x264-params", args.x264,
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-r", rate, "-movflags", "+faststart", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    start = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for buf in _ordered(pool, _frame, times, 4 * args.workers):
            enc.stdin.write(buf)
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                el = time.time() - start
                print(f"  {done}/{n} frames  {el:6.1f}s  {1000 * el / done:6.1f} ms/frame wall"
                      f"  eta {el / done * (n - done):5.0f}s", file=sys.stderr, flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    el = time.time() - start
    print(f"wrote {args.out}: {n} frames {w}x{h} @ {rate} fps in {el:.1f}s "
          f"({1000 * el / max(n, 1):.1f} ms/frame wall, {args.workers} workers)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stills", help="comma-separated seconds")
    ap.add_argument("--png-dir")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--timeline", default=DEFAULT_TIMELINE)
    ap.add_argument("--bench", type=int, help="time N single-process frames spread over the video")
    ap.add_argument("--bench-at", help="comma-separated seconds to time (with --bench 1)")
    ap.add_argument("--no-mb", action="store_true", help="disable motion blur")
    ap.add_argument("--x264", default=X264_PARAMS, help="libx264 parameters (default: %(default)s)")
    args = ap.parse_args()

    _init(args.timeline, args.scale, mb=not args.no_mb, fps=args.fps)
    dur = _show.duration

    if args.bench:
        import numpy as np
        ts = (np.array([float(x) for x in args.bench_at.split(",")]) if args.bench_at
              else np.linspace(0.5, dur - 3.0, args.bench))
        _frame(float(ts[0]))                       # warm caches
        per = []
        for t in ts:
            s = time.perf_counter()
            _frame(float(t))
            per.append(1000 * (time.perf_counter() - s))
        for t, ms in zip(ts, per):
            print(f"  t={t:7.2f}s  {ms:6.1f} ms")
        print(f"mean {sum(per) / len(per):.1f} ms/frame (single process, scale {args.scale})")
        return

    if args.stills:
        if not args.png_dir:
            ap.error("--stills needs --png-dir")
        os.makedirs(args.png_dir, exist_ok=True)
        jobs = [(float(s), os.path.join(args.png_dir, f"t{float(s):07.2f}.png"))
                for s in args.stills.split(",") if s.strip()]
        if len(jobs) == 1 or args.workers <= 1:
            out = [_png(j) for j in jobs]
        else:
            with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
                out = pool.map(_png, jobs)
        for p in out:
            print(p)
        return

    if not args.out:
        ap.error("--out is required (or use --stills / --bench)")
    fps = args.fps or _show.fps
    t0 = args.t0 if args.t0 is not None else 0.0
    t1 = args.t1 if args.t1 is not None else dur
    encode(args, fps, t0, t1)


if __name__ == "__main__":
    main()
