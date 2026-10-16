#!/usr/bin/env python3
"""Master music.wav -> audio.wav and (optionally) mux it with the silent picture.

    python3 mix.py                                  BUILD/audio/music.wav -> BUILD/audio/audio.wav
    python3 mix.py --mux VIDEO_IN.mp4 FINAL_OUT.mp4 ... then VIDEO_IN (video copied) + audio.wav -> FINAL_OUT
    python3 mix.py --selftest                       ... then mux a generated black dummy video of the exact
                                                    timeline length (BUILD/audio/test/) as a test

Mastering: ffmpeg loudnorm (pass 1) measures integrated loudness and true peak; one linear gain
takes the music to -18 LUFS; a 4x-oversampled look-ahead limiter (gain smoothed over ~40 ms,
multiplicative only) holds true peak under the ceiling where needed. Gain and limiter only
multiply, so digital silence (S14 window, the tail after S22 "black") stays exactly zero; no
dither is added. Output: 48 kHz stereo 24-bit PCM with exactly round(duration * 48000) samples.
Verification: ffmpeg ebur128 (true peak) on audio.wav; with --mux, ffprobe stream durations
(must match within one frame) and a decode-back of the AAC track (offset + silence check).
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import music  # noqa: E402  (timeline reader, BS.1770 + per-shot analysis)

SR = music.SR
TARGET_LUFS = -18.0
TP_MAX = -1.5            # required ceiling (dBTP)
TP_CEIL = -2.0           # limiter ceiling: 0.5 dB margin under the requirement


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p


def loudnorm_measure(path):
    """ffmpeg loudnorm first pass: returns its JSON (input_i, input_tp, input_lra, input_thresh)."""
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
               f"loudnorm=I={TARGET_LUFS}:TP={TP_MAX}:LRA=20:print_format=json", "-f", "null", "-"]).stderr
    return json.loads(err[err.rindex("{"):err.rindex("}") + 1])


def ebur128(path):
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex",
               "ebur128=peak=true:framelog=quiet", "-f", "null", "-"]).stderr
    summ = err[err.rindex("Summary:"):]
    get = lambda pat: float(re.search(pat, summ, re.S).group(1))
    return dict(I=get(r"I:\s+(-?[\d.]+) LUFS"), LRA=get(r"LRA:\s+(-?[\d.]+) LU"),
                TP=get(r"True peak:\s+Peak:\s+(-?[\d.inf]+) dBFS"))


def true_peak_per_sample(x, os_=4):
    """Per-sample peak of the 4x-oversampled signal (max over channels and the 4 sub-samples)."""
    up = signal.resample_poly(x, os_, 1, axis=1)
    return np.abs(up).max(0).reshape(-1, os_).max(1)[:x.shape[1]]


def limiter_gain(x, ceil_db, radius=0.04):
    """Look-ahead limiter gain <= ceiling/peak everywhere: min-filter over +-radius, then two box
    passes of total half-width radius/2 (each output is an average of minima that all cover the
    sample, so the gain never exceeds what that sample needs)."""
    p = true_peak_per_sample(x)
    g_raw = np.minimum(1.0, music.db(ceil_db) / np.maximum(p, 1e-12))
    if g_raw.min() >= 1.0:
        return np.ones(x.shape[1]), p
    r = int(radius * SR)
    g = ndimage.minimum_filter1d(g_raw, 2 * r + 1, mode="nearest")
    for _ in range(2):
        g = ndimage.uniform_filter1d(g, r // 2 + 1, mode="nearest")
    return np.minimum(g, g_raw), p


def write_wav24(path, x):
    q = np.clip(np.round(x.T * 8388607.0), -8388608, 8388607).astype("<i4")
    data = q.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    ch = x.shape[0]
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, SR, SR * ch * 3, ch * 3, 24))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def read_wav(path):
    sr, d = wavfile.read(path)
    if sr != SR:
        raise SystemExit(f"{path}: sample rate {sr}, expected {SR}")
    if d.dtype == np.int32:
        x = d.astype(np.float64) / 2 ** 31          # scipy returns 24-bit left-justified in int32
    elif d.dtype == np.int16:
        x = d.astype(np.float64) / 2 ** 15
    else:
        x = d.astype(np.float64)
    return x.T if x.ndim == 2 else np.vstack([x, x])


def master(music_wav, audio_wav, tl, log=print):
    n = int(round(tl.duration * SR))
    x = read_wav(music_wav)
    if x.shape[1] != n:
        log(f"  note: music.wav has {x.shape[1]} samples, timeline needs {n}: padding/trimming")
        x = np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))))[:, :n]
    zero_in = np.all(x == 0, axis=0)
    m = loudnorm_measure(music_wav)
    log(f"  pass 1 (ffmpeg loudnorm): I {m['input_i']} LUFS, TP {m['input_tp']} dBTP, "
        f"LRA {m['input_lra']} LU, threshold {m['input_thresh']}; numpy BS.1770 I {music.lufs(x):.2f}")
    gain_db = TARGET_LUFS - float(m["input_i"])
    for it in range(3):                                   # limiter steals a hair of loudness: re-trim
        y = x * music.db(gain_db)
        g, p = limiter_gain(y, TP_CEIL)
        z = y * g
        lu = music.lufs(z)
        if abs(lu - TARGET_LUFS) < 0.05:
            break
        gain_db += TARGET_LUFS - lu
    gr = 20 * np.log10(np.maximum(g, 1e-12))
    log(f"  gain {gain_db:+.2f} dB; "
        f"pre-limiter true peak (numpy 4x) {20 * np.log10(p.max()):.2f} dBTP; limiter: max gain reduction "
        f"{-gr.min():.2f} dB, active (< -0.1 dB) for {np.sum(gr < -0.1) / SR * 1000:.0f} ms total")
    assert np.all(z[:, zero_in] == 0), "digital silence was not preserved"
    write_wav24(audio_wav, z)
    return z


def report(audio_wav, tl, log=print):
    z = read_wav(audio_wav)
    n = int(round(tl.duration * SR))
    e = ebur128(audio_wav)
    log(f"  audio.wav: {z.shape[1]} samples ({z.shape[1] / SR:.4f}s; timeline {n} samples) "
        f"{'OK' if z.shape[1] == n else 'MISMATCH'}")
    log(f"  ffmpeg ebur128: integrated {e['I']:.1f} LUFS, LRA {e['LRA']:.1f} LU, true peak {e['TP']:.1f} dBTP "
        f"-> {'OK' if abs(e['I'] - TARGET_LUFS) <= 0.5 and e['TP'] <= TP_MAX else 'OUT OF SPEC'}")
    log(f"  numpy check: integrated {music.lufs(z):.2f} LUFS, sample peak {music.dbfs(np.max(np.abs(z))):.2f} dBFS")
    for line in music.format_shot_table(music.shot_table(z, tl)):
        log("  " + line)
    sil0, sil1 = tl.ev("S14", "silence")
    piano = tl.ev("S14", "piano")[0]
    b0, b1 = tl.ev("S22", "black")
    for name, (a, b) in (("S14 silence", (sil0, piano)), ("after S22 black", (b1, tl.duration))):
        st = music.window_stats(z, a, b)
        log(f"  {name} [{a:.3f}, {b:.3f}]: {st['zeros']}/{st['n']} samples exactly zero, peak {st['peak']:.1f} dBFS")
    return e


def ffprobe_streams(path):
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "stream=index,codec_type,codec_name,duration,nb_frames,sample_rate,channels,r_frame_rate,bit_rate"
               ":format=duration", "-of", "json", path]).stdout
    return json.loads(out)


def mux(video_in, audio_wav, out, tl, log=print):
    """Mux into a temporary file next to `out`; it is renamed to `out` only after every check
    passes, so a failed run never leaves (or overwrites) a final file."""
    d_, b_ = os.path.split(os.path.abspath(out))
    tmp = os.path.join(d_, f".{b_}.partial.mp4")
    ok = False
    try:
        ok = _mux_checked(video_in, audio_wav, tmp, tl, log)
    finally:
        if not ok:
            for p in (tmp, tmp + ".decoded.wav"):
                if os.path.exists(p):
                    os.remove(p)
    if not ok:
        raise SystemExit(f"audio/video durations differ by more than one frame: {out} was NOT written")
    os.replace(tmp, out)
    log(f"  -> {out}")
    return ok


def _mux_checked(video_in, audio_wav, out, tl, log=print):
    run(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-i", video_in, "-i", audio_wav,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
         "-movflags", "+faststart", "-f", "mp4", out])
    info = ffprobe_streams(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    vd, ad = float(v["duration"]), float(a["duration"])
    num, den = (int(k) for k in v["r_frame_rate"].split("/"))
    frame = den / num
    ok = abs(vd - ad) <= frame
    log(f"  muxed (temporary) {out}")
    log(f"  ffprobe: video {v['codec_name']} {vd:.4f}s ({v.get('nb_frames')} frames @ {num}/{den}), "
        f"audio {a['codec_name']} {a['sample_rate']} Hz {a['channels']} ch {int(a.get('bit_rate', 0)) // 1000} kb/s "
        f"{ad:.4f}s; |diff| {abs(vd - ad) * 1000:.1f} ms vs 1 frame {frame * 1000:.1f} ms -> {'OK' if ok else 'FAIL'}; "
        f"timeline {tl.duration:.4f}s")
    # decode the AAC back: alignment and the silence window survive the encode?
    dec = out + ".decoded.wav"
    run(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-i", out, "-map", "0:a:0", "-c:a", "pcm_f32le", dec])
    d = read_wav(dec)
    os.remove(dec)
    ref = read_wav(audio_wav)
    k = min(d.shape[1], ref.shape[1])
    a0 = int(tl.ev("S20", "merge")[0] * SR) - SR
    seg_r, seg_d = ref[0, a0:a0 + 2 * SR], d[0, a0 - 4800:a0 + 2 * SR + 4800]
    corr = signal.correlate(seg_d, seg_r, mode="valid")
    lag = int(np.argmax(corr)) - 4800
    sil0, piano = tl.ev("S14", "silence")[0], tl.ev("S14", "piano")[0]
    st = music.window_stats(d, sil0 + 0.025, piano - 0.025)
    log(f"  decoded AAC: {d.shape[1]} samples (wav {ref.shape[1]}), offset vs audio.wav {lag} samples; "
        f"S14 silence (inner, 25 ms guard for AAC frames) peak {st['peak']:.1f} dBFS, RMS {st['rms']:.1f} dBFS; "
        f"coding residual vs wav {music.dbfs(np.sqrt(np.mean((d[:, :k] - ref[:, :k]) ** 2))):.1f} dBFS RMS")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", default=os.path.join(music.BUILD, "timeline.json"))
    ap.add_argument("--music", default=os.path.join(music.AUDIO, "music.wav"))
    ap.add_argument("--audio", default=os.path.join(music.AUDIO, "audio.wav"))
    ap.add_argument("--mux", nargs=2, metavar=("VIDEO_IN", "FINAL_OUT"))
    ap.add_argument("--selftest", action="store_true", help="mux a generated black dummy video as a test")
    args = ap.parse_args()
    tl = music.Timeline(args.timeline)
    print(f"mix.py: {args.music} -> {args.audio} (timeline {tl.duration:.4f}s)")
    master(args.music, args.audio, tl)
    report(args.audio, tl)
    if args.selftest:
        tdir = os.path.join(music.AUDIO, "test")
        os.makedirs(tdir, exist_ok=True)
        dummy = os.path.join(tdir, "dummy_black.mp4")
        frames = int(round(tl.duration * tl.fps))
        run(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
             f"color=c=black:s=1920x1080:r={tl.fps}", "-frames:v", str(frames), "-c:v", "libx264",
             "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p", dummy])
        print(f"  dummy video {dummy}: {frames} frames")
        mux(dummy, args.audio, os.path.join(tdir, "test_mux.mp4"), tl)
    if args.mux:
        mux(args.mux[0], args.audio, args.mux[1], tl)


if __name__ == "__main__":
    main()
