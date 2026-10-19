#!/usr/bin/env python3
"""Original score + sound design for 「为什么网络让人变成二极管」, synthesized from scratch.

    python3 music.py              -> BUILD/audio/music.wav        48 kHz stereo float32, exact timeline length
                                     BUILD/audio/music_cues.json  every cue placed (absolute seconds)
                                     BUILD/audio/music_report.txt analysis (also printed)
                                     BUILD/audio/spectrogram.png  spectrogram + loudness strip + cue marks
    python3 music.py --no-check   -> render only
    python3 music.py --stems      -> also BUILD/audio/stems/{main,hold,accent,ticks}.wav

No samples, no downloaded audio: every sound is computed here with numpy/scipy
(additive felt-piano model, polyBLEP "analog" pad through a low-pass, sine drones,
inharmonic glass ticks, FFT convolution reverb with a synthetic stereo IR).
All cue times are read from BUILD/timeline.json at run time (events, shot bounds);
nothing here hardcodes an absolute time. Tonal centre D throughout: cold sections
lean D minor with a semitone rub, warm sections open to D major.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hk_sim  # noqa: E402  (only used to place stereo positions where the visuals sit)

BUILD = os.environ.get(
    "BUILD",
    "/tmp/claude-0/-home-user-english-words-ai-recommend/"
    "4af3582d-b5b8-57c7-bf82-51aca3b2d552/scratchpad/build",
)
AUDIO = os.path.join(BUILD, "audio")
SR = 48000
BEAT = 1.1           # the score's slow pulse in seconds; "half a beat" = BEAT / 2
PRE_LUFS = -20.0     # music.wav is pre-levelled here; mix.py makes the final -18 LUFS
ROOM_DBFS = -62.0    # S22 room-tone RMS in music.wav
TICK_D6, TICK_A5 = 86, 81   # glass tick pitches: warm sections / cold sections

# instrument base gains (linear) and reverb sends
PAD_GAIN, PAD_SEND = 0.115, 0.45
DRONE_GAIN = 0.045
PIANO_GAIN = 0.45
PULSE_GAIN = 0.040
TICK_GAIN = 0.030
LEAD_GAIN = 0.052
GRAIN_GAIN = 0.010
HOLD_GAIN = 0.032


# ----------------------------------------------------------------------------- helpers
def mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def db(x):
    return 10.0 ** (x / 20.0)


def pan_lr(p):
    a = (float(np.clip(p, -1, 1)) + 1.0) * np.pi / 4.0
    return np.cos(a), np.sin(a)


def smooth(u):
    """Raised-cosine ramp 0..1 (zero slope at both ends: no clicks)."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def bp_env(points, t):
    """Envelope through (time, value) points, raised-cosine between points.
    Two points at the same time make a step."""
    pts = sorted(points, key=lambda p: p[0])
    T = np.array([p[0] for p in pts], float)
    V = np.array([p[1] for p in pts], float)
    t = np.asarray(t, float)
    if len(T) == 1:
        return np.full(t.shape, V[0])
    i = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2)
    span = T[i + 1] - T[i]
    u = np.where(span > 0, (t - T[i]) / np.where(span > 0, span, 1.0), 1.0)
    out = V[i] + (V[i + 1] - V[i]) * smooth(u)
    out = np.where(t < T[0], V[0], out)
    return np.where(t >= T[-1], V[-1], out)


def dbpts(pts):
    """(time, dB or None) -> (time, linear); None = silent."""
    return [(t, 0.0 if v is None else db(v)) for t, v in pts]


# ----------------------------------------------------------------------------- timeline
class Timeline:
    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.path = path
        self.duration = float(d["duration"])
        self.fps = d["fps"]
        self.order = [s["id"] for s in d["shots"]]
        self.shots = {s["id"]: s for s in d["shots"]}

    def s(self, sid):
        return self.shots[sid]["start"]

    def e(self, sid):
        return self.shots[sid]["end"]

    def ev(self, sid, name):
        return tuple(self.shots[sid]["events"][name])

    def numbered(self, sid, prefix):
        out = []
        for k, (a, b) in self.shots[sid]["events"].items():
            if k.startswith(prefix + ":"):
                out.append((int(k.split(":")[1]), a, b))
        return sorted(out)


# ----------------------------------------------------------------------------- instruments
def polyblep_saw(f, ph0):
    """Band-limited sawtooth for a per-sample frequency array."""
    dt = f / SR
    ph = (ph0 + np.cumsum(dt)) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = ph > 1.0 - dt
    x = (ph[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


def pad_voice(midi, env_pts, rng):
    """Three slowly drifting detuned saws (spread L/C/R) + a sine body. Filtered later on the bus."""
    t0, t1 = env_pts[0][0], env_pts[-1][0]
    n = int(round((t1 - t0) * SR)) + 1
    t = np.arange(n) / SR
    env = bp_env([(a - t0, b) for a, b in env_pts], t)
    f0 = mtof(midi)
    out = np.zeros((2, n))
    for cents, pan in ((-6.5, -0.6), (0.4, 0.0), (7.0, 0.6)):
        r = rng.random(4)
        drift = (2.2 * np.sin(2 * np.pi * (0.05 + 0.06 * r[0]) * t + 2 * np.pi * r[1])
                 + 1.2 * np.sin(2 * np.pi * (0.17 + 0.10 * r[2]) * t + 2 * np.pi * r[3]))
        saw = polyblep_saw(f0 * 2.0 ** ((cents + drift) / 1200.0), rng.random())
        L, R = pan_lr(pan)
        out[0] += L * saw
        out[1] += R * saw
    out += 0.707 * 0.35 * np.sin(2 * np.pi * f0 * t + 2 * np.pi * rng.random())
    out *= 0.3 * env
    return t0, out


def piano_note(midi, vel, rng, off=None, rel=0.7):
    """Felt-piano-like note: slightly inharmonic partials, two detuned strings,
    double decay, soft felt attack. `off` (s after onset) damps it."""
    f0 = mtof(midi)
    t60 = float(np.interp(midi, [26, 50, 74, 98], [15.0, 9.0, 4.5, 1.8]))
    dur = t60 * 0.85 if off is None else min(t60 * 0.85, off + rel)
    n = int(dur * SR)
    t = np.arange(n) / SR
    B = float(np.interp(midi, [26, 60, 96], [0.00008, 0.00025, 0.0015]))
    tilt = 1.05 - 0.55 * vel
    det = 0.7e-3 if midi > 40 else 0.25e-3
    x = np.zeros(n)
    k = 1
    while k <= 16:
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk > 5200:
            break
        amp = np.exp(-(k - 1) * tilt) / np.sqrt(k) / (1.0 + (fk / 2800.0) ** 4)
        tau = t60 / 6.91 / (1.0 + 0.45 * (k - 1))
        env = 0.6 * np.exp(-t / (0.22 * tau)) + 0.4 * np.exp(-t / tau)
        ph = rng.random() * 2 * np.pi        # both strings struck by one hammer: start in phase
        x += amp * env * 0.5 * (np.sin(2 * np.pi * fk * (1 + det) * t + ph)
                                + np.sin(2 * np.pi * fk * (1 - det) * t + ph))
        k += 1
    x /= np.max(np.abs(x)) + 1e-12
    nt = int(0.06 * SR)                       # felt "thump": soft low-passed noise
    thump = signal.lfilter(*signal.butter(2, min(900.0, 4 * f0), fs=SR), rng.standard_normal(nt))
    thump *= np.exp(-t[:nt] / 0.012) * smooth(t[:nt] / 0.004)
    x[:nt] += 0.03 * vel * thump / (np.max(np.abs(thump)) + 1e-12)
    na = int((0.006 + 0.008 * (1 - vel)) * SR)
    x[:na] *= smooth(np.arange(na) / na)
    if off is not None:
        x *= 1.0 - smooth((t - off) / rel)
    nf = int(0.25 * SR)
    x[-nf:] *= 1.0 - smooth(np.arange(nf) / nf)
    return PIANO_GAIN * vel ** 1.3 * x


def lead_note(midi, dur, rng, att=0.18, rel=0.7, vib=7.0, glide_to=None, glide=(0.0, 0.0),
              swell=0.0, bright=1.0):
    """Soft sustained singing voice (sine + a few weak harmonics), delayed vibrato, optional glide."""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    cents = vib * smooth((t - 0.3) / 0.7) * np.sin(2 * np.pi * 4.7 * t + 2 * np.pi * rng.random())
    semis = np.zeros(n)
    if glide_to is not None:
        g0, g1 = glide
        semis = (glide_to - midi) * smooth((t - g0) / max(g1 - g0, 1e-3))
    f = mtof(midi) * 2.0 ** (semis / 12.0 + cents / 1200.0)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + bright * (0.16 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph) + 0.015 * np.sin(4 * ph))
    env = smooth(t / att) * (1.0 - smooth((t - dur) / rel))
    env *= (1.0 + swell * np.clip(t / max(dur, 1e-3), 0, 1)) / (1.0 + swell)
    return LEAD_GAIN * x * env


def pulse_note(midi, rng, level=1.0, att=0.07, decay=0.42, damp=None):
    """Low tonal pulse: soft swell then decay. Pitched, no pitch drop -> not a drum."""
    n = int(1.8 * SR)
    t = np.arange(n) / SR
    f = mtof(midi)
    x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t + rng.random() * 6.28)
         + 0.12 * np.sin(2 * np.pi * 3 * f * t + rng.random() * 6.28))
    env = smooth(t / att) * np.exp(-np.maximum(t - att, 0) / decay)
    if damp is not None:
        env *= 1.0 - smooth((t - damp) / 0.12)
    nf = int(0.1 * SR)
    env[-nf:] *= 1.0 - smooth(np.arange(nf) / nf)
    return PULSE_GAIN * level * x * env


TICK_PARTIALS = ((1.0, 1.0, 0.30), (2.0, 0.10, 0.12), (2.756, 0.30, 0.075), (5.404, 0.08, 0.03))


def tick(midi, rng, level=1.0, t_abs=None, decay=1.0):
    """Glassy tick: pure inharmonic partials (free bar ratios), 2 ms raised-cosine attack.
    With t_abs (onset in seconds) every partial's phase is locked to absolute time, so a tick
    always adds in phase to the still-ringing tail of an earlier tick of the same pitch (in the
    dense S20 run a random phase could half-cancel it). `decay` scales the ring time (dense ticks
    ring shorter so they neither pile up nor smear)."""
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    f0 = mtof(midi)
    x = np.zeros(n)
    for r, a, tau in TICK_PARTIALS:
        ph = rng.random() * 6.28 if t_abs is None else (2 * np.pi * f0 * r * (round(t_abs * SR) / SR)) % (2 * np.pi)
        x += a * np.exp(-t / (tau * decay)) * np.sin(2 * np.pi * f0 * r * t + ph)
    na = int(0.002 * SR)
    x[:na] *= smooth(np.arange(na) / na)
    nf = int(0.1 * SR)
    x[-nf:] *= 1.0 - smooth(np.arange(nf) / nf)
    return TICK_GAIN * level * x / np.max(np.abs(x))


def grain(midi, dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * mtof(midi) * t + rng.random() * 6.28) * np.hanning(n)


def make_ir(rng, length=4.2, predelay=0.024):
    """Synthetic stereo hall IR: decorrelated noise per channel, four bands with their own
    decay (highs die first), soft onset after a short pre-delay, unit energy per channel."""
    n = int(length * SR)
    t = np.arange(n) / SR
    lf = np.log(np.maximum(np.fft.rfftfreq(n, 1.0 / SR), 1.0))
    sig = lambda f: 1.0 / (1.0 + np.exp(-(lf - np.log(f)) / 0.18))
    bands = [(None, 350, 3.4), (350, 2500, 3.0), (2500, 6000, 1.9), (6000, None, 0.9)]
    ir = np.zeros((2, n))
    for ch in range(2):
        spec = np.fft.rfft(rng.standard_normal(n))
        for lo, hi, rt in bands:
            m = np.ones_like(lf)
            if lo:
                m *= sig(lo)
            if hi:
                m *= 1.0 - sig(hi)
            ir[ch] += np.fft.irfft(spec * m, n) * np.exp(-6.91 * t / rt)
    k = int(predelay * SR)
    ir = np.roll(ir, k, axis=1)
    ir[:, :k] = 0.0
    ko = int(0.02 * SR)
    ir[:, k:k + ko] *= smooth(np.arange(ko) / ko)
    kf = int(0.3 * SR)
    ir[:, -kf:] *= 1.0 - smooth(np.arange(kf) / kf)
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))


# ----------------------------------------------------------------------------- score container
class Score:
    BUSES = ("main", "hold", "accent")

    def __init__(self, n):
        self.n = n
        self.dry = {b: np.zeros((2, n)) for b in self.BUSES}
        self.send = {b: np.zeros(n) for b in self.BUSES}
        self.pad = np.zeros((2, n))
        self.tick_stem = np.zeros(n)
        self.pad_gain, self.pad_bright, self.gates = [], [], {}
        self.cues = {"ticks": [], "notes": [], "pulses": [], "windows": {}}

    def _span(self, t, length):
        i0 = int(round(t * SR))
        return i0, max(0, -i0), length - max(0, i0 + length - self.n)

    def add(self, x, t, pan=0.0, send=0.3, bus="main"):
        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            L, R = pan_lr(pan)
            x = np.vstack([x[0] * L, x[0] * R])
        i0, a, b = self._span(t, x.shape[1])
        if b <= a:
            return
        self.dry[bus][:, i0 + a:i0 + b] += x[:, a:b]
        self.send[bus][i0 + a:i0 + b] += send * 0.5 * (x[0, a:b] + x[1, a:b])

    def add_pad(self, x, t):
        i0, a, b = self._span(t, x.shape[1])
        if b > a:
            self.pad[:, i0 + a:i0 + b] += x[:, a:b]

    # convenience wrappers --------------------------------------------------
    def piano(self, t, midi, vel, rng, off=None, rel=0.7, send=0.35, bus="main", label=None, gain=1.0,
              pan=None):
        x = gain * piano_note(midi, vel, rng, off=None if off is None else off - t, rel=rel)
        self.add(x, t, pan=(midi - 64) / 45.0 if pan is None else pan, send=send, bus=bus)
        if label:
            self.cues["notes"].append({"name": label, "t": t, "midi": midi})

    def tick(self, t, midi, rng, level, shot, label, decay=1.0):
        x = tick(midi, rng, level, t_abs=t, decay=decay)
        self.add(x, t, pan=rng.uniform(-0.25, 0.25), send=0.6)
        i0, a, b = self._span(t, len(x))
        self.tick_stem[i0 + a:i0 + b] += x[a:b]
        self.cues["ticks"].append({"t": t, "f0": mtof(midi), "shot": shot, "event": label,
                                   "level": level})


def chord_seq(sc, rng, seq, end, rel=2.4):
    """Pad chords. seq = [(t, notes, att)], notes = midi or (midi, weight).
    A note held across consecutive chords keeps sounding (no re-attack); notes that
    leave cross-fade out while the new ones fade in."""
    chords = []
    for t, notes, att in seq:
        d = {}
        for m in notes:
            d[m if isinstance(m, int) else m[0]] = 1.0 if isinstance(m, int) else m[1]
        chords.append((t, d, att))
    runs, done = {}, []
    for i, (t, notes, att) in enumerate(chords):
        nxt = chords[i + 1][0] if i + 1 < len(chords) else end
        att = min(att, 0.9 * max(nxt - t, 0.05))
        for m in list(runs):
            if m not in notes:
                pts = runs.pop(m)
                done.append((m, pts + [(t, pts[-1][1]), (t + max(att, 0.8) * 1.2, 0.0)]))
        for m, w in notes.items():
            if m in runs:
                runs[m] += [(t, runs[m][-1][1]), (t + att, w)]
            else:
                runs[m] = [(t, 0.0), (t + att, w)]
    for m, pts in runs.items():
        done.append((m, pts + [(end, pts[-1][1]), (end + rel, 0.0)]))
    for m, pts in done:
        t0, x = pad_voice(m, pts, rng)
        sc.add_pad(x, t0)


def drone(sc, pts_db, rng, notes=((38, 1.0), (45, 0.32)), send=0.15):
    """Low sine drone with weak harmonics. L and R share each partial's frequency and phase
    except for a small fixed offset (+-0.25 rad) for width, so a mono fold-down never cancels it
    (no L/R detuning); the slow breathing is the 0.045 Hz amplitude swell."""
    pts = dbpts(pts_db)
    n0 = max(0, int(pts[0][0] * SR))
    n1 = min(sc.n, int(pts[-1][0] * SR))
    t = np.arange(n0, n1) / SR
    env = bp_env(pts, t) * (1.0 + 0.10 * np.sin(2 * np.pi * 0.045 * t + rng.random() * 6.28))
    x = np.zeros((2, len(t)))
    for m, w in notes:
        f = mtof(m)
        for h, a in ((1, 1.0), (2, 0.32), (3, 0.10)):
            ph = rng.random() * 6.28
            for ch, off in ((0, -0.25), (1, 0.25)):
                x[ch] += w * a * np.sin(2 * np.pi * h * f * t + ph + off)
    sc.add(DRONE_GAIN * 0.707 * x * env, n0 / SR, send=send)


def grain_cloud(sc, rng, t0, t1, density_pts, pitches, level=1.0, toward=None):
    """Sparse high grains. density_pts = (t, grains/s). toward = midi the pitches converge to."""
    t = t0
    while True:
        d = max(float(bp_env(density_pts, [t])[0]), 0.05)
        t += rng.exponential(1.0 / d)
        if t >= t1:
            break
        cand = pitches
        if toward is not None:
            u = (t - t0) / (t1 - t0)
            cand = [p for p in pitches if abs(p - toward) <= 12 * (1 - u) + 0.5] or [toward]
        g = grain(int(rng.choice(cand)), rng.uniform(0.04, 0.12), rng)
        sc.add(GRAIN_GAIN * level * rng.uniform(0.5, 1.0) * g, t, pan=rng.uniform(-0.7, 0.7), send=0.8)


def pulse_map(t0, beats, soft, freeze, p0=1.1, p1=0.9):
    """Tempo map for the S15-S17 pulse. Pulse k sounds where the phase function f(t) = k.
    Hard anchors: every beat in `beats` sits on a pulse (or, at a small cost, on an exact half
    pulse), and `freeze` sits exactly on a half pulse (half a beat after the last pulse).
    The pulse count between consecutive anchors is searched (multiples of 0.5 near what a gently
    tightening target period p0 -> p1 implies); f is a monotone cubic (PCHIP) through the anchors,
    so the tempo is continuous. The search prefers: tempo close to the target, tempo that only
    tightens, beats on whole pulses, and the `soft` times (S17 round starts) near a pulse or half
    pulse instead of just before/after one (a flam).
    Returns (pulse times in [t0, freeze), f, info)."""
    import itertools
    from scipy.interpolate import PchipInterpolator
    anchors = list(beats) + [freeze]
    span = max(freeze - beats[0], 1e-6)
    rate0 = lambda t: 1.0 / (p0 + (p1 - p0) * np.clip((t - beats[0]) / span, 0.0, 1.0))
    cands = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        tt = np.linspace(a, b, 2001)
        lo = np.floor(2.0 * np.trapezoid(rate0(tt), tt)) / 2.0
        cands.append([c for c in (lo - 0.5, lo, lo + 0.5, lo + 1.0) if c >= 0.5])
    pre_t = [t0 - 2.0, t0]
    pre_ph = [-(beats[0] - t) / p0 for t in pre_t]
    tt = np.arange(t0, freeze, 0.001)
    best = None
    for combo in itertools.product(*cands):
        ph = np.concatenate([[0.0], np.cumsum(combo)])
        if abs(ph[-1] % 1.0 - 0.5) > 1e-9:
            continue
        f = PchipInterpolator(pre_t + anchors, pre_ph + list(ph))
        r = f(tt, 1)
        cost = 400.0 * np.mean(np.log(r / rate0(tt)) ** 2)                 # near the target tempo
        cost += 50.0 * np.sum(np.maximum(-np.diff(r[::50]), 0.0))          # only tightens
        cost += 0.3 * sum(1 for p in ph[:-1] if p % 1.0)                   # beats on whole pulses
        fr = [float(f(t)) % 1.0 for t in soft]
        cost += 30.0 * sum(min(x, abs(x - 0.5), 1.0 - x) ** 2 for x in fr)  # rounds on the grid
        if best is None or cost < best[0]:
            best = (cost, combo, f, fr)
    _, combo, f, fr = best
    grid = np.arange(t0, freeze + 0.0005, 0.0005)
    fg = f(grid)
    out = []
    for k in range(int(np.ceil(fg[0])), int(np.floor(float(f(freeze)) - 1e-6)) + 1):
        tp = float(np.interp(k, fg, grid))
        for a in anchors:                                                   # exact on the anchors
            if abs(tp - a) < 1e-3:
                tp = a
        out.append(tp)
    per = 1.0 / f(tt, 1)
    info = dict(counts=[float(c) for c in combo], period_min=float(per.min()),
                period_max=float(per.max()), soft_phase=[round(x, 3) for x in fr],
                beat_phase=[round(float(f(b)) % 1.0, 3) for b in beats],
                freeze_phase=round(float(f(freeze)) % 1.0, 3))
    return out, f, info


def thin_ticks(starts, t_final, g0=0.5, g1=0.24, knee=0.65, slack=0.01):
    """S20: choose which of the accelerating round starts get a tick. The first start and t_final
    (the merge round, ticked by the caller) always tick, and the last ticked regular round sits
    within g1/2 of the last regular round. Target gap: shrinks geometrically from g0 to g1 over the
    first `knee` of the span, then holds at g1 (the rounds blur together faster than that; ticks at
    ~4 Hz stay soft and never rattle). Dynamic programming over (previous, current) pairs; a gap
    longer than the one before (beyond `slack`) is penalised hard. The final gap into t_final is the
    on-screen pause before the merge round and is exempt. Returns indices into starts."""
    pts = list(starts) + [t_final]
    n, last = len(pts), len(pts) - 1
    span = max(starts[-1] - pts[0], 1e-6)
    tgt = lambda t: g0 * (g1 / g0) ** np.clip((t - pts[0]) / (knee * span), 0.0, 1.0)

    def step(j, i):
        g = pts[i] - pts[j]
        if i == last:
            return 0.0 if pts[j] >= starts[-1] - 0.5 * g1 else np.inf
        if g < 0.7 * g1 or g > 1.3 * g0:
            return np.inf
        return 10.0 * np.log(g / tgt(0.5 * (pts[i] + pts[j]))) ** 2

    best = np.full((n, n), np.inf)
    back = np.full((n, n), -1, int)
    for i in range(1, n):
        best[i, 0] = step(0, i)
    for i in range(1, n):
        for j in range(1, i):
            c = step(j, i)
            if not np.isfinite(c):
                continue
            g = pts[i] - pts[j]
            for k in range(j):
                if not np.isfinite(best[j, k]):
                    continue
                gp = pts[j] - pts[k]
                pen = 0.0 if i == last else 100.0 * max(0.0, g - gp - slack) / gp
                if best[j, k] + c + pen < best[i, j]:
                    best[i, j], back[i, j] = best[j, k] + c + pen, k
    j = int(np.argmin(best[last]))
    if not np.isfinite(best[last, j]):                  # cannot happen with sane input; tick all
        return list(range(len(starts)))
    seq, i = [], last
    while j >= 0:
        seq.append(j)
        i, j = j, (back[i, j] if j > 0 else -1)
    return sorted(seq)


def screen_pan(x):
    """Opinion x in [0,1] -> stereo position (gentle, never hard-panned)."""
    return 0.6 * (2.0 * float(x) - 1.0)


# ----------------------------------------------------------------------------- the score
def compose(tl, sc, rng):
    s, e, ev = tl.s, tl.e, tl.ev
    D = tl.duration
    W = sc.cues["windows"]

    # --- cue times, all from the timeline -------------------------------------------
    card = ev("S04", "card")[0]
    pause0 = card - BEAT / 2                       # S04: the music stops for half a beat
    dim0, dim1 = ev("S04", "dim")
    lu0, lu1 = ev("S02", "light_up")
    rays0, rays1 = ev("S06", "rays")
    move0, move1 = ev("S06", "move")
    r9 = [a for _, a, _ in tl.numbered("S09", "round")]
    r4_0, r4_1 = ev("S10", "round:4")
    nar0, nar1 = ev("S11", "narrow")
    r12 = tl.numbered("S12", "round")
    ff0, ff1 = ev("S13", "fast_forward")
    sil0, sil1 = ev("S14", "silence")
    pno = ev("S14", "piano")[0]
    reset15 = ev("S15", "reset")[1]
    beats = [a for _, a, _ in tl.numbered("S16", "beat")]
    freeze = ev("S17", "freeze")[0]
    hl0, _ = ev("S18", "highlight")
    w0, w1 = ev("S18", "widen")
    b19, _ = ev("S19", "round:1")
    th0, th1 = ev("S19", "threads")
    merge = ev("S20", "merge")[0]
    black0, black1 = ev("S22", "black")
    card22 = ev("S22", "card")[0]
    W.update({"pause": [pause0, card], "s10_quiet": [e("S10") - 1.5, e("S10")], "fast_forward": [ff0, ff1],
              "s22_card": [card22, card22 + 1.0],
              "silence": [sil0, pno], "black": [black0, black1], "after_black": [black1, D],
              "freeze": freeze,
              "pulse_spans": [[s("S05"), move0], [reset15, beats[0]], [beats[0], beats[1]],
                              [beats[1], beats[2]], [beats[2], beats[3]], [beats[3], freeze]],
              "pulse_names": ["S05-06", "S15", "S16 beat1", "beat2", "beat3", "beat4-S17"]})

    # stereo positions taken from the simulation (where the agents sit on screen)
    demo = hk_sim.run(hk_sim.X0, 0.20)
    x6a, x6b = hk_sim.X0[5], demo[1][5]            # S05/S06: 6th rectangle, before/after one round
    left_c = float(np.mean(demo[-1][:hk_sim.N // 2]))   # S13 end state, left cluster centre

    # --- drone bed --------------------------------------------------------------------
    drone(sc, [
        (0.0, None), (s("S01") + 1.0, None), (s("S02") + 0.8, -15),        # S01: barely there
        (s("S03"), -12), (pause0 - 0.25, -12), (pause0, None),            # S04: stop
        (card + 0.9, None), (s("S05"), -9), (s("S07"), -9),
        (s("S08") + 1.5, -10), (r4_1, -10.5), (r4_1 + 2.0, None),        # S10: below the S20 merge
        (s("S11") + 0.2, None), (s("S11") + 3.0, -8), (ff0 - 1.0, -8), (ff0 - 0.1, None),
        (reset15 - 0.6, None), (reset15 + 2.5, -9), (freeze, -9), (freeze + 3.5, -12),
        (s("S19"), -11), (merge, -9), (merge + 3.0, -10), (e("S21") - 4.0, -12),
        (s("S22") - 0.2, None), (D, None)], rng)                          # gone before S22

    # --- pad --------------------------------------------------------------------------
    D3, A3, E4 = 50, 57, 64
    # S02-S04: open D(sus2), dark; it stops with everything else for the half beat
    chord_seq(sc, rng, [(s("S02") + 0.2, [D3, A3, E4], 3.0)], end=pause0 - 0.2, rel=0.2)
    # after the card -> S05..S07 neutral, S08 warm D(add9), S09 rising, S10 resolve
    chord_seq(sc, rng, [
        (card + 1.2, [D3, A3, E4], 3.0),
        (move1, [D3, A3, E4, (66, 0.8)], 1.5),                       # S06: colour leans gold
        (s("S08"), [38, D3, A3, E4, 66, 69], 2.2),                   # D add9, open
        (r9[0], [38, D3, 55, 62, 69, 71], 1.6),                      # G/D        top B4
        (r9[1], [47, 54, A3, 62, 66, 73], 1.6),                      # Bm(add9)   top C#5
        (r9[2], [45, 52, A3, 62, 69, 76], 1.6),                      # Asus4      top E5
        (r4_0, [45, 52, A3, 61, 69, 76], 0.9),                       # A          (sus resolves)
        (r4_1, [38, D3, A3, 62, 66, 69, 74, 78], 0.5),               # D          top F#5: resolution
    ], end=r4_1 + 0.5, rel=1.8)
    # S12: cold D minor; a Bb3 rubs against A3 from round 3
    chord_seq(sc, rng, [
        (s("S12") - 0.5, [D3, 53, A3], 2.5),
        (r12[2][1], [D3, 53, A3, (58, 0.7)], 1.5),
    ], end=ff0 - 0.9, rel=0.8)
    # S15-S17: Dm(add9) cold; each S16 beat narrows the voicing; it splits into E4/F4 on beat 4
    chord_seq(sc, rng, [
        (reset15, [45, D3, 53, 57, E4], 2.5),
        (beats[1], [D3, 53, 57, E4], 0.6),
        (beats[2], [53, 57, E4], 0.3),
        (beats[3], [53, E4, (65, 0.8)], 0.6),
    ], end=freeze + 0.2, rel=3.5)
    # S18-S22: open fifth -> warming duet harmony -> suspended A11 -> D major at merge -> recede
    chord_seq(sc, rng, [
        (s("S18") - 0.5, [D3, A3], 3.0),
        (b19 + 0.1, [43, D3, 59, 62, 66], 1.6),                      # Gmaj7
        (b19 + 1.2, [42, D3, A3, 62, 66], 1.2),                      # D/F#
        (th0 + 0.3, [49, 52, A3, 59, E4], 1.2),                      # A(add9)/C#
        (th1 + 0.2, [D3, A3, 62, 66], 1.2),                          # D
        (e("S19") - 2.5, [43, D3, A3, 59, 66], 1.8),                 # Gmaj9
        (s("S20") + 0.3, [45, 55, 59, 62, E4], 2.0),                 # A11 (G/A), held, crescendo
        (merge, [(38, 0.8), (45, 0.7), D3, A3, 62, 66, (69, 0.8), (74, 0.7)], 0.4),  # D major, open
        (s("S21") + 1.0, [38, D3, A3, E4, (66, 0.8)], 2.0),          # recede to Dadd9
    ], end=e("S21") - 1.8, rel=2.2)                                   # faded out by the S22 card

    sc.pad_gain = dbpts([
        (0.0, -10), (s("S03"), -9), (dim0, -9), (dim1, -12), (card + 1.0, -11), (s("S05"), -9),
        (s("S08"), -8), (s("S08") + 2.0, -6), (r9[2], -6.5), (r4_1, -8.5), (s("S11"), -8.5), (s("S12"), -6),
        (s("S15"), -8), (beats[3], -7), (freeze, -7), (s("S18"), -10), (s("S19"), -7),
        (s("S20"), -8.5), (s("S20") + 0.55 * (merge - s("S20")), -7.5), (merge - 0.8, -6),  # very slow
        (merge, -4.5), (merge + 1.5, -4.5), (e("S21"), -9.5), (D, -11)])
    bright = [(0.0, 0.15), (s("S03"), 0.15), (e("S03"), 0.28), (s("S05"), 0.22), (s("S08"), 0.25),
              (s("S08") + 2.5, 0.45), (r9[2], 0.55), (r4_1, 0.65), (r4_1 + 3.0, 0.5),
              (s("S12"), 0.06), (ff0, 0.05), (s("S15"), 0.14)]
    level = 0.14
    for b, new in zip(beats, (0.14, 0.10, 0.06, 0.03)):              # one notch narrower per beat
        bright += [(b, level), (b + 0.5, new)]
        level = new
    bright += [(freeze, level), (s("S18"), 0.18), (s("S19"), 0.25), (b19 + 1.0, 0.38),
               (s("S20"), 0.36), (merge, 0.6), (merge + 1.5, 0.58), (e("S21"), 0.25), (D, 0.15)]
    sc.pad_bright = bright

    # --- S02-S04: sparse felt piano, grains, the half-beat stop, one low note ----------
    for u, m, v in ((0.02, 62, 0.27), (0.30, 69, 0.25), (0.58, 66, 0.24), (0.86, 64, 0.23)):
        sc.piano(lu0 + u * (lu1 - lu0), m, v, rng, off=pause0, gain=0.78)
    d3 = e("S03") - s("S03")
    for u, m, v in ((0.10, 71, 0.25), (0.36, 69, 0.24), (0.62, 74, 0.25), (0.88, 66, 0.23)):
        sc.piano(s("S03") + u * d3, m, v, rng, off=pause0, gain=0.85)
    sc.piano(dim0, 64, 0.23, rng, off=pause0, gain=0.7)
    grain_cloud(sc, rng, s("S03") + 0.3, pause0 - 0.3,
                [(s("S03") + 0.3, 0.5), (s("S03") + 2.0, 4.0), (e("S03") - 1.0, 5.0),
                 (e("S03") + 0.5, 1.5), (pause0, 0.3)],
                [74, 76, 78, 81, 83, 86, 88, 90])
    # centred (the card sits mid-screen) and restrained: about the level of the music around it
    sc.piano(card, 38, 0.42, rng, off=s("S06"), rel=2.5, bus="accent", label="S04 card low note",
             gain=0.5, pan=0.0, send=0.15)
    n = int(5.0 * SR)
    t = np.arange(n) / SR
    sub = 0.010 * np.sin(2 * np.pi * mtof(38) * t) * smooth(t / 0.02) * np.exp(-t / 1.6)
    sub[-int(0.3 * SR):] *= 1 - smooth(np.arange(int(0.3 * SR)) / (0.3 * SR))
    sc.add(sub, card, send=0.1, bus="accent")

    # --- S05/S06: slow focused pulse; converging grains; one rising note -----------------
    times = np.arange(s("S05") + 0.5, move0 - 0.2, BEAT)
    for i, tp in enumerate(times):
        lv = min(1.0, (i + 1) / 3.0, (len(times) - i) / 4.0)
        sc.add(pulse_note(38, rng, 0.9 * lv), tp, send=0.25)
        sc.cues["pulses"].append(float(tp))
    grain_cloud(sc, rng, rays0, rays1, [(rays0, 6.0), (rays1, 6.0)], [69, 71, 74, 76, 78, 81, 83, 86, 88, 93],
                level=0.9, toward=81)
    g_len = move1 - move0
    x = lead_note(69, g_len + 0.9, rng, att=0.35, rel=1.4, vib=4.0, glide_to=74, glide=(0.3, 0.3 + g_len))
    L = np.linspace(screen_pan(x6a), screen_pan(x6b), x.size)            # follows the rectangle
    a = np.vstack([x * np.cos((L + 1) * np.pi / 4), x * np.sin((L + 1) * np.pi / 4)])
    sc.add(0.6 * a, move0 - 0.3, send=0.45)
    sc.piano(move1, 74, 0.22, rng, gain=0.7)

    # --- ticks for every round / beat ----------------------------------------------------
    for sid, midi, lv in (("S07", TICK_D6, 0.8), ("S09", TICK_D6, 1.1), ("S10", TICK_D6, 1.1),
                          ("S12", TICK_A5, 0.75), ("S17", TICK_A5, 0.75), ("S19", TICK_D6, 0.8)):
        for k, a0, _ in tl.numbered(sid, "round"):
            sc.tick(a0, midi, rng, lv, sid, f"round:{k}")
    for k, a0, _ in tl.numbered("S16", "beat"):
        sc.tick(a0, TICK_A5, rng, 0.85, "S16", f"beat:{k}")
    # S20: accelerating rounds -> thin + soften the ticks; the ticked rounds are chosen so the
    # gaps shrink steadily (~0.5 s -> the final gap before the merge round), like the rounds do
    r20 = tl.numbered("S20", "round")
    reg, (kf, af, _) = r20[:-1], r20[-1]
    kmin, kmax = reg[0][0], reg[-1][0]
    sel = thin_ticks([a0 for _, a0, _ in reg], af)
    nxt = [reg[j][1] for j in sel[1:]] + [af]
    for i, tn in zip(sel, nxt):
        k, a0, _ = reg[i]
        u = (k - kmin) / max(kmax - kmin, 1)
        sc.tick(a0, TICK_D6, rng, 0.65 - 0.3 * u, "S20", f"round:{k}",
                decay=float(np.clip((tn - a0) / 0.6, 0.4, 1.0)))
    sc.tick(af, TICK_D6, rng, 0.6, "S20", f"round:{kf}")

    # --- S10: the warm chord resolves at the end of round 4 -----------------------------
    for i, m in enumerate((50, 57, 66, 74)):
        sc.piano(r4_1 + 0.06 * i, m, 0.21, rng, off=r4_1 + 1.2, rel=1.0,
                 label="S10 resolution" if i == 0 else None)

    # --- S11: only the bass; a faint high tone slides down a third as the bracket narrows --
    x = lead_note(81, (nar1 - nar0) + 1.2, rng, att=0.6, rel=1.5, vib=0.0, glide_to=77,
                  glide=(0.6, 0.6 + (nar1 - nar0)), bright=0.5)
    sc.add(0.42 * x, nar0 - 0.6, send=0.5)

    # --- S12 -> S14: the single long tone (own bus), thinning before the silence ----------
    h0 = r12[-1][1] - 0.5
    h_end = e("S13")
    hn = int((sil0 - h0) * SR)
    t = np.arange(hn) / SR + h0
    f = mtof(62)
    fund = bp_env([(h0, 0.0), (h0 + 1.6, 1.0), (h_end + 2.5, 1.0), (sil0 - 0.4, 0.0)], t)
    harm = bp_env([(h0, 0.0), (h0 + 1.6, 1.0), (h_end + 0.8, 1.0), (h_end + 5.0, 0.0)], t)
    ph = 2 * np.pi * f * (t - h0)
    x = HOLD_GAIN * (fund * np.sin(ph) + harm * (0.22 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)))
    sc.add(x, h0, send=0.25, bus="hold")
    # 「极轻」: soft velocity (darker), well under the music level, centred on the black gap
    sc.piano(pno, 74, 0.16, rng, gain=0.5, send=0.5, pan=0.0, label="S14 piano")

    # --- S15-S17: the pulse returns, tightens slightly, stops on a half beat at freeze ----
    # every S16 beat lands on a pulse; freeze is exactly half a pulse after the last one
    r17 = [a for _, a, _ in tl.numbered("S17", "round")]
    pulses, _, pinfo = pulse_map(reset15, beats, r17, freeze, p0=BEAT, p1=0.9)
    sc.cues["pulse_map"] = pinfo
    lvl = [(reset15, 0.3), (reset15 + 3.0, 1.0), (beats[3], 1.1), (freeze, 1.15)]
    dec = [(reset15, 0.42), (beats[0], 0.42), (beats[3], 0.33), (freeze, 0.30)]   # tighter articulation
    for tp in pulses:
        if tp >= reset15 + 0.3:
            sc.add(pulse_note(38, rng, float(bp_env(lvl, [tp])[0]), decay=float(bp_env(dec, [tp])[0])),
                   tp, send=0.25)
            sc.cues["pulses"].append(float(tp))
    sc.add(pulse_note(38, rng, 0.9, damp=0.25), freeze, send=0.25)
    sc.cues["notes"].append({"name": "S17 last pulse (half beat)", "t": freeze, "midi": 38})
    sc.cues["pulses"].append(float(freeze))

    # --- S18: one lonely high line; S19: a second voice; S20: held, swelling; merge ----------
    seg = w1 - w0
    mel = [(hl0 + 0.05, 78), (hl0 + 1.15, 76), (w0, 74), (w0 + 0.25 * seg, 76),
           (w0 + 0.5 * seg, 78), (w0 + 0.75 * seg, 81)]
    ends = [m[0] for m in mel[1:]] + [e("S18") + 0.6]
    for (t0, m), t1 in zip(mel, ends):
        sc.add(0.55 * lead_note(m, t1 - t0 + 0.12, rng), t0, pan=screen_pan(left_c), send=0.5)
        sc.piano(t0, m, 0.14, rng, send=0.45)
    top = [(b19 + 0.1, 79), (b19 + 1.2, 78), (th0 + 0.3, 76), (th1 + 0.2, 78), (e("S19") - 2.5, 81)]
    low = [(b19 + 0.1, 71), (b19 + 1.2, 69), (th0 + 0.3, 73), (th1 + 0.2, 74), (e("S19") - 2.5, 78)]
    for voice, pan, g in ((top, 0.0, 0.7), (low, 0.3, 0.5)):
        ends = [m[0] for m in voice[1:]] + [s("S20") + 0.4]
        for (t0, m), t1 in zip(voice, ends):
            sc.add(g * lead_note(m, t1 - t0 + 0.12, rng), t0, pan=pan, send=0.5)
    for m, m2, pan, g in ((76, 78, 0.0, 0.8), (71, 69, 0.3, 0.55)):    # E5/B4 -> F#5/A4
        sc.add(g * lead_note(m, merge - s("S20") - 0.15, rng, att=1.2, swell=0.6), s("S20") + 0.3,
               pan=pan, send=0.5)
        sc.add(g * lead_note(m2, 3.5, rng, att=0.6, rel=2.5), merge, pan=pan, send=0.5)
    for i, m in enumerate((50, 57, 66, 74)):                          # the pad carries the low D
        sc.piano(merge + 0.07 * i, m, 0.25, rng, off=s("S22") + 1.0, rel=1.5,
                 label="S20 merge chord" if i == 0 else None)

    # --- gates (applied after reverb, so tails cannot leak into the stops) ---------------
    sc.gates["main"] = [(0.0, 1.0), (pause0 - 0.18, 1.0), (pause0, 0.0), (card + 1.0, 0.0),
                        (card + 2.4, 1.0), (ff0 - 0.6, 1.0), (ff0, 0.0), (sil0, 0.0), (sil0, 1.0)]
    sc.gates["hold"] = [(0.0, 1.0), (sil0 - 0.6, 1.0), (sil0, 0.0), (D, 0.0)]
    sc.gates["master"] = [(0.0, 1.0), (sil0 - 0.4, 1.0), (sil0, 0.0), (pno, 0.0), (pno, 1.0),
                          (black0, 1.0), (black1, 0.0), (D, 0.0)]
    sc.room = [(s("S22") + 0.5, 0.0), (s("S22") + 2.5, 1.0), (black0, 1.0), (black1, 0.0)]


# ----------------------------------------------------------------------------- render
def lufs(x):
    """ITU-R BS.1770-4 integrated loudness of a (2, n) array at 48 kHz."""
    p = kweighted_power(x)
    return gated_lufs(p)


def kweighted_power(x):
    y = signal.lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                       [1.0, -1.69065929318241, 0.73248077421585], x, axis=-1)
    y = signal.lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], y, axis=-1)
    return np.sum(y ** 2, axis=0)


def block_ms(p, blk=0.4, hop=0.1):
    nb, nh = int(blk * SR), int(hop * SR)
    if len(p) < nb:
        return np.array([])
    c = np.concatenate([[0.0], np.cumsum(p)])
    i = np.arange(0, len(p) - nb + 1, nh)
    return (c[i + nb] - c[i]) / nb


def gated_lufs(p):
    ms = block_ms(p)
    if ms.size == 0:
        return -np.inf
    L = -0.691 + 10 * np.log10(ms + 1e-30)
    ms = ms[L > -70]
    if ms.size == 0:
        return -np.inf
    rel = -0.691 + 10 * np.log10(np.mean(ms)) - 10
    ms = ms[-0.691 + 10 * np.log10(ms) > rel]
    return -0.691 + 10 * np.log10(np.mean(ms))


def render(tl, seed=11, stems_dir=None, log=print):
    n = int(round(tl.duration * SR))
    rng = np.random.default_rng(seed)
    sc = Score(n)
    t_start = time.time()
    compose(tl, sc, rng)
    log(f"  composed in {time.time() - t_start:.1f}s")
    ta = np.arange(n) / SR

    sc.pad = signal.sosfilt(signal.butter(2, 85, "highpass", fs=SR, output="sos"), sc.pad, axis=1)
    dark = signal.sosfilt(signal.butter(2, 620, fs=SR, output="sos"), sc.pad, axis=1)
    brt = signal.sosfilt(signal.butter(2, 2000, fs=SR, output="sos"), sc.pad, axis=1)
    b = bp_env(sc.pad_bright, ta)
    pad = (dark * (1 - b) + brt * b) * bp_env(sc.pad_gain, ta) * PAD_GAIN
    del dark, brt
    sc.dry["main"] += pad
    sc.send["main"] += PAD_SEND * 0.5 * (pad[0] + pad[1])
    del pad

    ir = make_ir(np.random.default_rng(seed + 1))
    hp = signal.butter(2, 22, "highpass", fs=SR, output="sos")
    lp = signal.butter(2, 9500, fs=SR, output="sos")
    out = np.zeros((2, n))
    for name in Score.BUSES:
        y = sc.dry[name].copy()
        nz = np.nonzero(sc.send[name])[0]
        if nz.size:
            a, z = nz[0], min(n, nz[-1] + ir.shape[1])
            for c in range(2):
                y[c, a:z] += signal.oaconvolve(sc.send[name][a:z], ir[c])[:z - a]
        y = signal.sosfilt(lp, signal.sosfilt(hp, y, axis=1), axis=1)   # DC/rumble out, no harsh top
        if name in sc.gates:
            y *= bp_env(sc.gates[name], ta)
        sc.dry[name] = y
        out += y
    gate = bp_env(sc.gates["master"], ta)
    out *= gate
    gain = db(PRE_LUFS - lufs(out))
    out *= gain
    room = np.random.default_rng(seed + 2).standard_normal((2, n))
    room = signal.lfilter([1.0], [1.0, -0.995], room, axis=1)
    room = signal.sosfilt(signal.butter(2, [30, 500], "bandpass", fs=SR, output="sos"), room, axis=1)
    room /= np.sqrt(np.mean(room ** 2))
    out += db(ROOM_DBFS) * room * bp_env(sc.room, ta)
    out *= gate                                   # exact digital zero in the silence / after black
    log(f"  rendered in {time.time() - t_start:.1f}s (pre-level gain {20 * np.log10(gain):+.1f} dB)")
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
        for name in Score.BUSES:
            wavfile.write(os.path.join(stems_dir, f"{name}.wav"), SR,
                          (sc.dry[name] * gate * gain).T.astype(np.float32))
        wavfile.write(os.path.join(stems_dir, "ticks.wav"), SR, (sc.tick_stem * gain).astype(np.float32))
    sc.cues["gain_db"] = 20 * np.log10(gain)
    return out, sc.tick_stem * gain, sc.cues


# ----------------------------------------------------------------------------- analysis
def dbfs(v):
    return 20 * np.log10(max(float(v), 1e-12))


def shot_table(x, tl):
    p = kweighted_power(x)
    rows = []
    for sid in tl.order:
        a, b = int(round(tl.s(sid) * SR)), int(round(tl.e(sid) * SR))
        seg = x[:, a:b]
        rows.append((sid, tl.s(sid), tl.e(sid), dbfs(np.max(np.abs(seg))),
                     dbfs(np.sqrt(np.mean(seg ** 2))), gated_lufs(p[a:b])))
    return rows


def format_shot_table(rows):
    out = ["shot   start    end     peak dBFS  RMS dBFS  LUFS(shot)"]
    for sid, a, b, pk, rms, lu in rows:
        out.append(f"{sid}  {a:7.2f} {b:7.2f}   {pk:8.1f}  {rms:8.1f}   {lu:8.1f}")
    return out


def window_stats(x, t0, t1):
    a, b = int(np.ceil(t0 * SR)), int(np.floor(t1 * SR))
    seg = x[:, a:b]
    if seg.size == 0:
        return dict(n=0, peak=-np.inf, rms=-np.inf, zeros=0)
    return dict(n=seg.shape[1], peak=dbfs(np.max(np.abs(seg))), rms=dbfs(np.sqrt(np.mean(seg ** 2))),
                zeros=int(np.sum(np.all(seg == 0, axis=0))))


def flux_onset(mono, t, f0, pre=0.3, post=0.1, nfft=512, hop=24):
    """Onset of a tick in the full mix. STFT (10.7 ms Hann, 0.5 ms hop); magnitudes at the tick's
    four partials, each bin whitened by its median over the 0.3 s before the event; the summed
    magnitude m(t) of a sinusoid switched on at t0 rises symmetrically around t0, so the maximum
    of m(t + 6 ms) - m(t - 6 ms) estimates the onset. Returns (onset, that maximum / 95th
    percentile of |rise| before the event) - the second value says how clearly it stands out."""
    a, b = int((t - pre - 0.02) * SR), int((t + post + 0.02) * SR)
    seg = mono[a:b]
    win = np.hanning(nfft)
    c = np.arange(nfft // 2, len(seg) - nfft // 2, hop)
    X = np.abs(np.fft.rfft(np.stack([seg[k - nfft // 2:k + nfft // 2] for k in c]) * win, axis=1))
    bins = []
    for ratio, _, _ in TICK_PARTIALS:
        k = int(round(ratio * f0 / (SR / nfft)))
        bins += list(range(k - 1, k + 2))
    tc = a / SR + c / SR
    pre_frames = (tc >= t - pre) & (tc < t - 0.05)
    Xb = X[:, bins] / (np.median(X[pre_frames][:, bins], axis=0) + 1e-9)   # whiten: quiet bins count more
    m = Xb.sum(1)
    lag = int(round(0.006 * SR / hop))
    rise = np.full(m.shape, -np.inf)
    rise[lag:-lag] = m[2 * lag:] - m[:-2 * lag]      # m(t + 6 ms) - m(t - 6 ms): peaks at a step onset
    reg = (tc >= t - 0.04) & (tc <= t + 0.04)
    i = np.argmax(np.where(reg, rise, -np.inf))
    bg = np.percentile(np.abs(rise[pre_frames & np.isfinite(rise)]), 95)
    return tc[i], rise[i] / (bg + 1e-12)


def detect_ticks(x, tick_stem, cues):
    mono = x.mean(0)
    rows = []
    for tk in cues["ticks"]:
        t, f0 = tk["t"], tk["f0"]
        mix_on, prom = flux_onset(mono, t, f0)
        # dry tick stem: first sample that rises 10% of the way from the preceding tail to the peak
        a = int((t - 0.03) * SR)
        st = np.abs(tick_stem[a:int((t + 0.03) * SR)])
        base = st[:int(0.028 * SR)].max()
        pk = st[int(0.03 * SR):].max()
        i = np.nonzero(st > base + 0.1 * (pk - base))[0]
        stem_on = a / SR + i[0] / SR if i.size else np.nan
        rows.append(dict(tk, mix_err_ms=(mix_on - t) * 1000, stem_err_ms=(stem_on - t) * 1000,
                         prominence=prom, peak_dbfs=dbfs(pk)))
    return rows


def click_scan(x):
    hf = signal.sosfilt(signal.butter(4, 11000, "highpass", fs=SR, output="sos"), x, axis=1)
    fr = 96
    m = (x.shape[1] // fr) * fr
    e = np.sqrt(np.mean(hf[:, :m].reshape(2, -1, fr) ** 2, axis=2)).max(0)
    edb = 20 * np.log10(e + 1e-12)
    med = ndimage.median_filter(edb, 251)
    flag = (edb > -70) & (edb > med + 15)
    jump = np.max(np.abs(np.diff(x, axis=1)))
    return dict(frames=int(flag.sum()), times=[round(i * fr / SR, 3) for i in np.nonzero(flag)[0][:10]],
                hf_max_dbfs=float(edb.max()), max_step=float(jump))


def spectrogram_png(x, tl, cues, path, width=2400):
    from PIL import Image, ImageDraw, ImageFont
    H, HL, top = 560, 150, 28
    mono = x.mean(0)
    n = mono.size
    nfft = 4096
    win = np.hanning(nfft)
    centers = np.linspace(0, n - 1, width * 2).astype(int)
    pad = np.concatenate([np.zeros(nfft // 2), mono, np.zeros(nfft // 2)])
    frames = np.stack([pad[c:c + nfft] for c in centers]) * win
    mag = np.abs(np.fft.rfft(frames, axis=1)) / (win.sum() / 2)
    mag = np.maximum(mag[0::2], mag[1::2])                     # 2 frames per column, keep max
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    rowf = 30 * (16000 / 30) ** (1 - (np.arange(H) + 0.5) / H)
    edges = 30 * (16000 / 30) ** (1 - np.arange(H + 1) / H)
    img = np.zeros((H, width))
    binw = SR / nfft
    for r in range(H):
        sel = (freqs <= edges[r]) & (freqs > edges[r + 1])
        if sel.any():
            img[r] = mag[:, sel].max(1)
        else:                                   # row narrower than a bin: interpolate
            fb = rowf[r] / binw
            i0 = int(fb)
            img[r] = mag[:, i0] * (1 - (fb - i0)) + mag[:, i0 + 1] * (fb - i0)
    dbimg = 20 * np.log10(img + 1e-12)
    anchors = [(-120, (0, 0, 0)), (-100, (18, 10, 52)), (-80, (90, 26, 110)), (-60, (200, 70, 60)),
               (-40, (248, 180, 80)), (-20, (255, 250, 225))]
    rgb = np.zeros((H, width, 3))
    for c in range(3):
        rgb[..., c] = np.interp(dbimg, [a for a, _ in anchors], [col[c] for _, col in anchors])
    im = Image.new("RGB", (width, top + H + HL + 24), (8, 8, 8))
    im.paste(Image.fromarray(rgb.astype(np.uint8)), (0, top))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    X = lambda t: int(t / tl.duration * (width - 1))
    for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        y = top + int(np.log(16000 / f) / np.log(16000 / 30) * H)
        d.line([(0, y), (8, y)], fill=(160, 160, 160))
        d.text((10, y - 7), f"{f}Hz" if f < 1000 else f"{f // 1000}k", fill=(160, 160, 160), font=font)
    for sid in tl.order:
        xs = X(tl.s(sid))
        d.line([(xs, top), (xs, top + H + HL)], fill=(90, 90, 90))
        d.text((xs + 3, 6), sid, fill=(220, 220, 220), font=font)
    for tk in cues["ticks"]:
        d.line([(X(tk["t"]), top), (X(tk["t"]), top + 10)], fill=(90, 220, 255))
    for nm in cues["notes"]:
        d.line([(X(nm["t"]), top + H), (X(nm["t"]), top + H - 14)], fill=(120, 255, 120))
    w = cues["windows"]
    d.rectangle([X(w["silence"][0]), top + 2, X(w["silence"][1]), top + H - 2], outline=(0, 255, 0))
    d.rectangle([X(w["pause"][0]), top + 2, max(X(w["pause"][1]), X(w["pause"][0]) + 1), top + H - 2],
                outline=(0, 255, 0))
    d.rectangle([X(w["after_black"][0]), top + 2, X(w["after_black"][1]), top + H - 2], outline=(0, 255, 0))
    # momentary loudness strip (400 ms), -70..-10 LUFS
    ms = block_ms(kweighted_power(x), 0.4, 0.05)
    Lm = -0.691 + 10 * np.log10(ms + 1e-30)
    tt = np.arange(len(Lm)) * 0.05 + 0.2
    y0 = top + H
    Y = lambda L: y0 + int((1 - (np.clip(L, -70, -10) + 70) / 60) * (HL - 4)) + 2
    for g, col in ((-18, (120, 100, 40)), (-30, (60, 60, 60)), (-50, (60, 60, 60))):
        d.line([(0, Y(g)), (width, Y(g))], fill=col)
        d.text((width - 70, Y(g) - 14), f"{g} LUFS", fill=(150, 150, 150), font=font)
    pts = [(X(t), Y(L)) for t, L in zip(tt, Lm)]
    d.line(pts, fill=(232, 192, 102), width=1)
    d.text((6, top + H + HL + 4), "momentary loudness (400 ms) | cyan = ticks, green = notes/silence windows",
           fill=(170, 170, 170), font=font)
    im.save(path)


def analyze(x, tl, cues, tick_stem=None, png=None, title="music.wav"):
    L = []
    n = x.shape[1]
    L.append(f"== analysis of {title} ==")
    L.append(f"samples {n} = {n / SR:.4f}s (timeline {tl.duration:.4f}s -> {int(round(tl.duration * SR))})")
    L.append(f"overall: peak {dbfs(np.max(np.abs(x))):.1f} dBFS, RMS {dbfs(np.sqrt(np.mean(x ** 2))):.1f} dBFS, "
             f"integrated {lufs(x):.2f} LUFS (BS.1770 numpy)")
    dc = x.mean(1)
    kern = np.ones(SR) / SR
    mov = max(np.max(np.abs(np.convolve(x[c, ::10], kern[::10] * 10, mode="valid"))) for c in range(2))
    L.append(f"DC: mean L {dc[0]:.2e}, R {dc[1]:.2e}; worst 1-s moving mean {mov:.2e}")
    L += format_shot_table(shot_table(x, tl))
    w = cues["windows"]
    for key in ("pause", "silence", "after_black", "s10_quiet", "s22_card"):
        st = window_stats(x, *w[key])
        L.append(f"window {key:12s} [{w[key][0]:.3f}, {w[key][1]:.3f}]: {st['n']} samples, "
                 f"{st['zeros']} exactly zero, peak {st['peak']:.1f} dBFS, RMS {st['rms']:.1f} dBFS")
    a, b = w["fast_forward"]
    seg = x[:, int(a * SR):int(b * SR)]
    blk = int(0.1 * SR)
    m = (seg.shape[1] // blk) * blk
    r = 20 * np.log10(np.sqrt(np.mean(seg[:, :m].reshape(2, -1, blk) ** 2, axis=(0, 2))) + 1e-12)
    L.append(f"S13 fast_forward steadiness: 100-ms RMS range {r.min():.2f} .. {r.max():.2f} dBFS "
             f"(spread {r.max() - r.min():.2f} dB)")
    if tick_stem is not None:
        rows = detect_ticks(x, tick_stem, cues)
        me = np.array([abs(r_["mix_err_ms"]) for r_ in rows])
        se = np.array([abs(r_["stem_err_ms"]) for r_ in rows])
        L.append(f"ticks: {len(rows)} placed; onset error vs event: mix (spectral flux) max {np.nanmax(me):.2f} ms "
                 f"(mean {np.nanmean(me):.2f}), dry stem max {np.nanmax(se):.2f} ms; "
                 f"flux prominence min {min(r_['prominence'] for r_ in rows):.1f}x")
        for r_ in rows:
            L.append(f"  {r_['shot']} {r_['event']:9s} t={r_['t']:8.3f}  mix {r_['mix_err_ms']:+6.2f} ms  "
                     f"stem {r_['stem_err_ms']:+5.2f} ms  peak {r_['peak_dbfs']:6.1f} dBFS  "
                     f"prominence {r_['prominence']:6.1f}x")
    pl = np.array(cues.get("pulses", []))
    if pl.size > 2 and "freeze" in cues["windows"]:
        W = cues["windows"]
        per = []
        for name, (a, b) in zip(W["pulse_names"], W["pulse_spans"]):
            q = pl[(pl >= a) & (pl < b)]
            if q.size > 1:
                per.append(f"{name} {np.mean(np.diff(q)):.3f}s")
        fz = W["freeze"]
        bf = pl[pl < fz - 1e-6]
        L.append(f"pulse: {pl.size} pulses; mean period " + ", ".join(per)
                 + f"; freeze {fz:.3f} = last beat {bf[-1]:.3f} + {fz - bf[-1]:.3f}s "
                 f"(= {(fz - bf[-1]) / (bf[-1] - bf[-2]):.2f} beat), one damped pulse on it, then none")
        if "pulse_map" in cues:
            pm = cues["pulse_map"]
            L.append(f"pulse map: pulses per anchor interval {pm['counts']}, period range "
                     f"{pm['period_min']:.3f}..{pm['period_max']:.3f}s, freeze phase {pm['freeze_phase']}")
        for tk in cues["ticks"]:
            if tk["shot"] in ("S16", "S17"):
                prev, nxt = pl[pl <= tk["t"] + 1e-6], pl[pl > tk["t"] + 1e-6]
                if prev.size and nxt.size:
                    L.append(f"  {tk['shot']} {tk['event']:8s} t={tk['t']:8.3f}: nearest pulse "
                             f"{(pl[np.argmin(np.abs(pl - tk['t']))] - tk['t']) * 1000:+7.1f} ms, phase in pulse "
                             f"cycle {(tk['t'] - prev[-1]) / (nxt[0] - prev[-1]):.2f}")
    t20 = [tk["t"] for tk in cues["ticks"] if tk["shot"] == "S20"]
    if len(t20) > 1:
        L.append(f"S20 ticks: {len(t20)}; intervals " + " ".join(f"{d:.3f}" for d in np.diff(t20)))
    for nm in cues["notes"]:
        i = int(np.ceil(nm["t"] * SR))
        seg = np.abs(x[:, i:i + int(0.08 * SR)]).max(0)
        on = np.nonzero(seg > 0.05 * seg.max())[0]
        L.append(f"note '{nm['name']}' at {nm['t']:.3f}: reaches 5% of its 80-ms peak after "
                 f"{on[0] / SR * 1000 if on.size else float('nan'):.1f} ms")
    c = click_scan(x)
    L.append(f"click scan (>11 kHz bursts 15 dB over local median): {c['frames']} frames {c['times']}; "
             f"HF max {c['hf_max_dbfs']:.1f} dBFS; largest sample step {c['max_step']:.4f}")
    spec = np.abs(np.fft.rfft(x.mean(0)[::1])) ** 2
    fr = np.fft.rfftfreq(n, 1 / SR)
    L.append(f"energy above 6 kHz: {10 * np.log10(spec[fr > 6000].sum() / spec.sum()):.1f} dB of total; "
             f"above 10 kHz: {10 * np.log10(spec[fr > 10000].sum() / spec.sum() + 1e-30):.1f} dB")
    if png:
        spectrogram_png(x, tl, cues, png)
        L.append(f"spectrogram -> {png}")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", default=os.path.join(BUILD, "timeline.json"))
    ap.add_argument("--out", default=os.path.join(AUDIO, "music.wav"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    tl = Timeline(args.timeline)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"music.py: timeline {args.timeline} ({tl.duration:.3f}s)")
    out, tick_stem, cues = render(tl, args.seed, os.path.join(AUDIO, "stems") if args.stems else None)
    wavfile.write(args.out, SR, out.T.astype(np.float32))
    cues_path = os.path.join(os.path.dirname(args.out), "music_cues.json")
    with open(cues_path, "w") as f:
        json.dump(cues, f, indent=1)
    print(f"  wrote {args.out} and {cues_path}")
    if args.no_check:
        return
    sr, data = wavfile.read(args.out)
    x = data.T.astype(np.float64)
    assert sr == SR
    lines = analyze(x, tl, cues, tick_stem, os.path.join(os.path.dirname(args.out), "spectrogram.png"))
    with open(os.path.join(os.path.dirname(args.out), "music_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
