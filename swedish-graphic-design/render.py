"""Render the film: soundtrack + frames -> H.264 / AAC MP4.

    python3 render.py                       # full film -> swedish_graphic_design.mp4
    python3 render.py --start 0 --end 20    # a section, for checking
    python3 render.py --workers 4 --crf 20

Frames are drawn with pycairo and piped as raw BGRx into ffmpeg. Each
worker renders a contiguous range of frames into its own segment; the
segments are joined without re-encoding and muxed with the soundtrack.
"""

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process

from sgd.core import FPS, W, H, NFRAMES

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")


def encoder_cmd(path, crf, preset):
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr0", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,setsar=1",
        "-c:v", "libx264", "-preset", preset, "-tune", "animation", "-crf", str(crf),
        "-profile:v", "high", "-level", "4.1",
        "-x264-params", "keyint=300:min-keyint=30:scenecut=40:aq-mode=3",
        "-threads", "2",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-r", str(FPS), "-an", path,
    ]


def render_segment(index, f0, f1, crf, preset):
    from sgd.film import Renderer
    path = os.path.join(BUILD, f"seg_{index:02d}.mp4")
    proc = subprocess.Popen(encoder_cmd(path, crf, preset), stdin=subprocess.PIPE)
    r = Renderer()
    t0 = time.time()
    for f in range(f0, f1):
        buf = r.render(f / FPS)
        proc.stdin.write(buf.data)
        if (f - f0) % 300 == 0:
            el = time.time() - t0
            print(f"  [seg {index}] frame {f} ({f - f0}/{f1 - f0})  {el:.0f}s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for segment {index}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=NFRAMES / FPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--out", default=os.path.join(HERE, "swedish_graphic_design.mp4"))
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)

    f_start = int(round(args.start * FPS))
    f_end = int(round(args.end * FPS))
    t_begin = time.time()

    wav = os.path.join(BUILD, "soundtrack.wav")
    if not args.no_audio:
        print("Synthesising soundtrack ...", flush=True)
        from sgd import audio
        audio.build(wav)

    n = f_end - f_start
    k = max(1, min(args.workers, n))
    bounds = [f_start + n * i // k for i in range(k + 1)]
    print(f"Rendering frames {f_start}..{f_end - 1} with {k} workers ...", flush=True)
    procs = []
    for i in range(k):
        p = Process(target=render_segment, args=(i, bounds[i], bounds[i + 1], args.crf, args.preset))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            sys.exit("a render worker failed")

    listing = os.path.join(BUILD, "segments.txt")
    with open(listing, "w") as fh:
        for i in range(k):
            fh.write(f"file 'seg_{i:02d}.mp4'\n")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", listing]
    if not args.no_audio:
        cmd += ["-ss", f"{f_start / FPS:.6f}", "-t", f"{n / FPS:.6f}", "-i", wav,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "256k", "-ar", "48000"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", "-shortest", args.out]
    subprocess.run(cmd, check=True)
    for i in range(k):
        os.remove(os.path.join(BUILD, f"seg_{i:02d}.mp4"))
    print(f"Done in {time.time() - t_begin:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
