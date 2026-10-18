"""Original soundtrack: minimal electronic, 96 BPM, 180 s, 48 kHz stereo.

A slow chord bed in D Dorian / F major, a soft sine-kick pulse, filtered
noise hats and clicks, a sine bass and a mallet-like FM arpeggio. Accents
come from the scenes' own EVENTS lists, so every sound lands on the frame
whose motion it belongs to. The disruption (Scene 07) drops the pulse for a
dissonant drone and a single low hit. Everything is synthesised here from
a fixed seed; no samples are used.
"""

import numpy as np
from scipy import signal

from .core import BAR, BEAT, DURATION

SR = 48000
N = int(DURATION * SR)
RNG = np.random.default_rng(1996)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def s_(sec):
    return int(round(sec * SR))


# --------------------------------------------------------------------------
# Harmony: one chord per bar (2.5 s). (bass note, pad voicing)
# --------------------------------------------------------------------------
CHORDS = {
    "Dm9": (38, (53, 57, 60, 64)),
    "Bbmaj9": (34, (53, 57, 60, 62)),
    "F/A": (33, (53, 57, 60, 64)),
    "Fmaj9": (41, (52, 55, 57, 60)),
    "Csus2": (36, (55, 60, 62, 67)),
    "C69": (36, (52, 57, 62, 67)),
    "Gm9": (31, (53, 58, 62, 69)),
    "A7sus4": (33, (55, 57, 62, 64)),
    "C/E": (40, (55, 60, 62, 64)),
    "Dm7": (38, (53, 57, 60, 62)),
    "Am7": (33, (52, 55, 60, 64)),
    "Gm7": (31, (53, 58, 62, 65)),
    "Csus4": (36, (53, 55, 60, 65)),
    "C": (36, (52, 55, 60, 64)),
    "Tension": (38, (50, 51, 57, 58)),
}

CHART = (
    # Scene 01  0-20 s
    "Dm9", "Dm9", "Dm9", "Dm9", "Bbmaj9", "Fmaj9", "C69", "Dm9",
    # Scene 02  20-45 s
    "Dm9", "Bbmaj9", "F/A", "Csus2", "Dm9", "Bbmaj9", "Gm9", "A7sus4", "Dm9", "C69",
    # Scene 03  45-70 s
    "Fmaj9", "C/E", "Dm7", "Bbmaj9", "Fmaj9", "C/E", "Gm7", "Bbmaj9", "Csus4", "C",
    # Scene 04  70-95 s
    "Fmaj9", "Am7", "Bbmaj9", "C69", "Dm9", "Dm9", "Gm9", "C69", "F/A", "Bbmaj9",
    # Scene 05  95-120 s
    "Dm9", "Dm9", "Bbmaj9", "Bbmaj9", "Fmaj9", "C/E", "Dm9", "Bbmaj9", "Gm7", "Csus4",
    # Scene 06  120-145 s
    "Fmaj9", "C/E", "Dm9", "Bbmaj9", "Fmaj9", "C/E", "Dm9", "Bbmaj9", "Gm9", "Csus4",
    # Scene 07  145-165 s
    "Fmaj9", "C/E", "Tension", "Tension", "Bbmaj9", "F/A", "Gm9", "Csus4",
    # Scene 08  165-180 s
    "Bbmaj9", "C69", "Fmaj9", "Dm9", "Bbmaj9", "Fmaj9",
)
assert len(CHART) == 72


def bars(*ranges):
    out = set()
    for a, b in ranges:
        out.update(range(a, b + 1))
    return out


# Arrangement (inclusive bar ranges)
KICK_13 = bars((8, 31), (34, 37), (40, 59), (62, 64))
KICK_1 = bars((4, 7), (32, 33), (38, 39), (65, 65), (66, 70))
HATS = bars((10, 31), (34, 37), (40, 59), (62, 64))
GHOST = bars((18, 27), (48, 57))
SNARE = bars((42, 47), (48, 57))
BASS = bars((8, 31), (34, 37), (40, 59), (62, 65), (66, 70))
ARP = bars((10, 17), (22, 31), (34, 37), (42, 47), (48, 57), (62, 64))
ARP_PATTERNS = {
    2: ((0, 1, 2, 3, 2, 1, 2, 3), 0.5),
    3: ((0, 2, 1, 3), 0.25),
    4: ((0, 3, 1, 2, 3, 1, 2, 0), 0.5),
    5: ((0, 2, 3, 1, 2, 3, 0, 2), 0.5),
    6: ((0, 1, 2, 3, 4, 3, 2, 1), 0.5),
    7: ((0, 2, 4, 2), 1.0),
}


def scene_of_bar(b):
    edges = (8, 18, 28, 38, 48, 58, 66, 72)
    for k, e in enumerate(edges):
        if b < e:
            return k + 1
    return 8


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class Bus:
    def __init__(self):
        self.l = np.zeros(N + SR * 6)
        self.r = np.zeros(N + SR * 6)

    def add(self, start, sig, gain=1.0, pan=0.0):
        """Add a mono or (2, n) signal at time start (s). pan -1..1."""
        i = s_(start)
        if i < 0:
            sig = sig[..., -i:]
            i = 0
        if sig.ndim == 1:
            n = min(len(sig), len(self.l) - i)
            gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
            gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
            self.l[i:i + n] += sig[:n] * gain * gl
            self.r[i:i + n] += sig[:n] * gain * gr
        else:
            n = min(sig.shape[1], len(self.l) - i)
            self.l[i:i + n] += sig[0, :n] * gain
            self.r[i:i + n] += sig[1, :n] * gain


def lowpass(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), "low")
    return signal.lfilter(b, a, x)


def highpass(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), "high")
    return signal.lfilter(b, a, x)


def bandpass(x, f0, f1, order=2):
    b, a = signal.butter(order, (f0 / (SR / 2), f1 / (SR / 2)), "band")
    return signal.lfilter(b, a, x)


def adsr(n, a, r, total=None):
    """Attack / release envelope over n samples (linear-in-dB style curves)."""
    t = np.arange(n) / SR
    dur = n / SR if total is None else total
    env = np.minimum(1.0, t / max(a, 1e-4))
    env = 0.5 - 0.5 * np.cos(np.pi * env)
    rel = np.clip((dur - t) / max(r, 1e-4), 0, 1)
    return env * (0.5 - 0.5 * np.cos(np.pi * rel))


def soft_tone(f, t, bright=0.0):
    """Sine with a touch of odd harmonics (a rounded triangle)."""
    ph = 2 * np.pi * f * t
    return np.sin(ph) + bright * (np.sin(3 * ph) / 9 + np.sin(5 * ph) / 25)


def fm(f, dur, ratio=2.0, index=1.5, idecay=0.08, adecay=0.5, attack=0.003):
    t = tt(dur)
    mod = index * np.exp(-t / idecay) * np.sin(2 * np.pi * f * ratio * t)
    env = np.minimum(1, t / attack) * np.exp(-t / adecay)
    return np.sin(2 * np.pi * f * t + mod) * env


def noise(n):
    return RNG.standard_normal(n)


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

def pad(music):
    """Chord bed. Each pitch is held for as long as consecutive chords
    contain it, so common tones sustain and only moving voices change."""
    level = np.ones(72)
    level[0:4] = (0.55, 0.65, 0.72, 0.8)
    level[32:34] = 0.62                     # neutral plate: let the red dot speak
    level[38:40] = 0.66                     # typographic zoom
    level[60:62] = 0.0                      # the tension drone replaces it
    level[71] = 0.9
    runs = {}
    for b, name in enumerate(CHART):
        if level[b] == 0:
            continue
        for m in CHORDS[name][1]:
            runs.setdefault(m, []).append(b)
    events = []
    for m, bl in runs.items():
        start = bl[0]
        prev = bl[0]
        for b in bl[1:] + [None]:
            if b is not None and b == prev + 1:
                prev = b
                continue
            events.append((m, start, prev))
            if b is not None:
                start = prev = b
    for m, b0, b1 in events:
        t0 = b0 * BAR + (1.25 if b0 == 0 else 0.0)
        t1 = (b1 + 1) * BAR
        dur = t1 - t0 + 1.6
        t = tt(dur)
        f = mtof(m)
        g = np.interp(t + t0, np.arange(72) * BAR + BAR / 2, level)
        a = 1.4 if b0 == 0 else 0.7
        env = adsr(len(t), a, 1.6) * g
        for det, pan in ((-0.0028, -0.6), (0.0, 0.0), (0.0031, 0.6)):
            sig = soft_tone(f * (1 + det), t + RNG.random(), 0.35) * env
            music.add(t0, sig, 0.030, pan)
        if m == max(CHORDS[CHART[b0]][1]):
            shimmer = np.sin(2 * np.pi * 2 * f * (t + RNG.random())) * env
            music.add(t0, shimmer, 0.011, 0.3)


def drums(dr):
    kick = None
    t = tt(0.45)
    f = 46 + 64 * np.exp(-t / 0.032)
    ph = 2 * np.pi * np.cumsum(f) / SR
    kick = np.sin(ph) * np.exp(-t / 0.19) * np.minimum(1, t / 0.002)
    kick = lowpass(kick, 900)
    hat_n = s_(0.09)
    for b in range(72):
        t0 = b * BAR
        sc = scene_of_bar(b)
        if b in KICK_13 or b in KICK_1:
            beats = (0, 2) if b in KICK_13 else (0,)
            soft = 0.75 if b in KICK_1 else 1.0
            if sc == 8:
                soft = 0.6
            for k in beats:
                dr.add(t0 + k * BEAT, kick, 0.27 * soft)
        if b in HATS:
            for k in range(4):
                h = bandpass(noise(hat_n), 6500, 11500) * np.exp(-np.arange(hat_n) / SR / 0.028)
                dr.add(t0 + (k + 0.5) * BEAT, h, 0.036 * (1.0 if k % 2 else 0.8), pan=0.25)
        if b in GHOST:
            for k in range(16):
                if k % 2 == 0:
                    continue
                h = bandpass(noise(hat_n), 7500, 12000) * np.exp(-np.arange(hat_n) / SR / 0.010)
                dr.add(t0 + k * BEAT / 4, h, 0.018, pan=-0.3)
        if b in SNARE:
            for k in (1, 3):
                n = s_(0.25)
                tn = np.arange(n) / SR
                sn = bandpass(noise(n), 900, 4500) * np.exp(-tn / 0.07)
                sn += 0.6 * np.sin(2 * np.pi * 185 * tn) * np.exp(-tn / 0.05)
                dr.add(t0 + k * BEAT, sn, 0.028, pan=0.1)


def bass(bs):
    for b in range(72):
        if b not in BASS:
            continue
        root = CHORDS[CHART[b]][0]
        sc = scene_of_bar(b)
        pattern = ((0, 1.9), (2, 1.9)) if sc != 8 else ((0, 3.8),)
        if sc in (3, 6):
            pattern = ((0, 1.4), (1.5, 0.45), (2, 1.9))
        for beat, length in pattern:
            dur = length * BEAT + 0.12
            t = tt(dur)
            f = mtof(root)
            sig = np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
            env = adsr(len(t), 0.015, 0.12) * np.exp(-t / 2.2)
            bs.add(b * BAR + beat * BEAT, lowpass(sig * env, 420), 0.115)


def arps(ar):
    for b in range(72):
        if b not in ARP:
            continue
        sc = scene_of_bar(b)
        pat, step = ARP_PATTERNS[sc]
        notes = list(CHORDS[CHART[b]][1]) + [CHORDS[CHART[b]][1][0] + 12]
        n_steps = int(round(4 / step))
        for k in range(n_steps):
            idx = pat[k % len(pat)]
            m = notes[idx % len(notes)] + 12
            accent = 1.0 if (k * step) % 1 == 0 else 0.72
            vel = accent * (0.8 if sc == 3 else 1.0)
            sig = fm(mtof(m), 1.1, ratio=2.0, index=1.2, idecay=0.07, adecay=0.38 if sc != 3 else 0.2)
            pan = 0.35 * np.sin(k * 1.7)
            ar.add(b * BAR + k * step * BEAT, sig, 0.066 * vel, pan)


def tension(fx):
    """Scene 07, bars 60-61: two detuned low voices a semitone apart and a
    faint high cluster, swelling into the cut."""
    t0 = 60 * BAR
    dur = 2 * BAR + 0.5
    t = tt(dur)
    env = adsr(len(t), 1.2, 0.6) * (0.7 + 0.3 * np.minimum(1, t / (dur * 0.8)))
    low = np.zeros(len(t))
    for m in (38, 39):
        f = mtof(m)
        for h in range(1, 9):
            low += np.sin(2 * np.pi * f * h * t + h) / h
    low = lowpass(low, 520) * env
    fx.add(t0, low, 0.060, -0.2)
    hi = (np.sin(2 * np.pi * mtof(69) * t) + np.sin(2 * np.pi * mtof(70) * t)) * env
    fx.add(t0, hi, 0.012, 0.3)
    # Rising filtered noise into the cut at bar 60 beat 3.
    rd = 1.9 * BEAT
    tr = tt(rd)
    riser = bandpass(noise(len(tr)), 300, 2500) * (tr / rd) ** 2.5
    fx.add(t0 + 3 * BEAT - rd, riser, 0.05)


# --------------------------------------------------------------------------
# Synchronised accents
# --------------------------------------------------------------------------

def ev_sound(kind, p):
    """Return (signal, gain, reverb send, pan) for one event."""
    if kind == "bloom":
        t = tt(3.5)
        f = mtof(p.get("note", 50))
        env = (1 - np.exp(-t / 0.18)) * np.exp(-t / 1.3)
        sig = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.5)) * env
        return sig, 0.15, 0.5, 0.0
    if kind == "glide":
        dur = p.get("dur", 1.0)
        t = tt(dur + 0.6)
        f = mtof(76) * 2 ** (0.5 * np.minimum(t / dur, 1) / 12 * 7)
        ph = 2 * np.pi * np.cumsum(f) / SR
        env = np.sin(np.pi * np.clip(t / (dur + 0.6), 0, 1)) ** 2
        return np.sin(ph) * env, 0.030, 0.7, 0.2
    if kind == "thud":
        t = tt(0.5)
        f = 50 + 40 * np.exp(-t / 0.04)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16) * np.minimum(1, t / 0.003)
        return sig, 0.17, 0.15, 0.0
    if kind in ("click", "tick"):
        t = tt(0.08)
        f = (2600 if kind == "click" else 3400) * p.get("pitch", 1.0)
        sig = np.sin(2 * np.pi * f * t) * np.exp(-t / (0.012 if kind == "click" else 0.006))
        return sig, 0.09 if kind == "click" else 0.06, 0.3, RNG.uniform(-0.4, 0.4)
    if kind == "ticks":
        n, step = p.get("n", 4), p.get("step", 0.06)
        t = tt(n * step + 0.1)
        sig = np.zeros(len(t))
        for k in range(n):
            i = s_(k * step)
            tk = t[: len(t) - i]
            f = 3000 * (1 + 0.04 * ((k * 7) % 5))
            sig[i:] += np.sin(2 * np.pi * f * tk) * np.exp(-tk / 0.005) * (0.8 + 0.2 * (k % 2))
        return sig, 0.05, 0.3, 0.1
    if kind == "snap":
        t = tt(0.15)
        f = 1700 * p.get("pitch", 1.0)
        sig = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.018)
        sig += 0.5 * bandpass(noise(len(t)), 3000, 9000) * np.exp(-t / 0.012)
        return sig, 0.085, 0.25, RNG.uniform(-0.3, 0.3)
    if kind == "low":
        t = tt(3.0)
        f = mtof(p.get("note", 36))
        env = np.minimum(1, t / 0.02) * np.exp(-t / 1.1)
        sig = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.4)) * env
        sig += 0.25 * lowpass(noise(len(t)), 300) * np.exp(-t / 0.25)
        return sig, 0.19, 0.35, 0.0
    if kind == "sub":
        t = tt(3.0)
        f = mtof(p.get("note", 36)) * (1 + 0.15 * np.exp(-t / 0.08))
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.01) * np.exp(-t / 1.4)
        return sig, 0.18, 0.25, 0.0
    if kind == "chime":
        f = mtof(p.get("note", 76))
        sig = fm(f, 3.5, ratio=3.5, index=0.9, idecay=0.5, adecay=1.1)
        sig += 0.4 * fm(f * 2, 3.5, ratio=1.0, index=0.2, idecay=0.3, adecay=0.6)
        return sig, 0.055, 0.6, RNG.uniform(-0.3, 0.3)
    if kind == "bell":
        f = mtof(p.get("note", 84))
        sig = fm(f, 4.0, ratio=1.41, index=1.4, idecay=0.6, adecay=1.3)
        return sig, 0.050, 0.65, 0.15
    if kind == "swell":
        dur = p.get("dur", 1.0)
        t = tt(dur + 0.4)
        env = np.clip(t / dur, 0, 1) ** 2.2 * np.exp(-np.clip(t - dur, 0, None) / 0.12)
        sig = bandpass(noise(len(t)), 400, 3200) * env
        return sig, 0.035, 0.6, 0.0
    if kind == "hit":
        t = tt(4.0)
        f = 41 + 30 * np.exp(-t / 0.05)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 1.0) * np.minimum(1, t / 0.002)
        sig += 0.6 * lowpass(noise(len(t)), 900) * np.exp(-t / 0.3)
        sig += 0.3 * bandpass(noise(len(t)), 2000, 8000) * np.exp(-t / 0.03)
        return sig, 0.24, 0.55, 0.0
    if kind == "tension":
        return None
    raise KeyError(kind)


def accents(fx, events):
    for t0, kind, p in events:
        snd = ev_sound(kind, p)
        if snd is None:
            continue
        sig, gain, send, pan = snd
        fx.add(t0, sig, gain, pan)
        fx.sends.append((t0, sig * gain, send, pan))


# --------------------------------------------------------------------------
# Space and master
# --------------------------------------------------------------------------

def reverb_ir(seconds=3.4, tau=0.42):
    t = tt(seconds)
    out = []
    for _ in range(2):
        bright = highpass(noise(len(t)), 1500) * np.exp(-t / (tau * 0.45))
        dark = lowpass(noise(len(t)), 2200) * np.exp(-t / tau)
        ir = 0.35 * bright + dark
        ir[: s_(0.022)] = 0
        out.append(ir / np.sqrt(np.sum(ir ** 2)))
    return out


def stereo_delay(bus, l_time, r_time, fb=0.35, mix=0.35):
    """Ping-pong style echo: two taps per side."""
    out_l = bus.l.copy()
    out_r = bus.r.copy()
    dl, dr_ = s_(l_time), s_(r_time)
    mono = 0.5 * (bus.l + bus.r)
    out_l[dl:] += mix * mono[:-dl]
    out_r[dr_:] += mix * 0.85 * mono[:-dr_]
    out_l[dl + dr_:] += mix * fb * mono[:-(dl + dr_)]
    out_r[2 * dr_:] += mix * fb * 0.8 * mono[:-2 * dr_]
    bus.l, bus.r = out_l, out_r


def render(events):
    music, dr, bs, ar, fx = Bus(), Bus(), Bus(), Bus(), Bus()
    fx.sends = []
    pad(music)
    drums(dr)
    bass(bs)
    arps(ar)
    stereo_delay(ar, 0.3125, 0.46875)
    tension(fx)
    accents(fx, events)

    # Reverb send: pad and arps generously, accents per event.
    send = Bus()
    send.l += 0.30 * music.l + 0.35 * ar.l + 0.06 * dr.l
    send.r += 0.30 * music.r + 0.35 * ar.r + 0.06 * dr.r
    for t0, sig, amt, pan in fx.sends:
        send.add(t0, sig, amt, pan)
    ir_l, ir_r = reverb_ir()
    wet_l = signal.fftconvolve(send.l, ir_l)[: len(send.l)]
    wet_r = signal.fftconvolve(send.r, ir_r)[: len(send.r)]

    left = music.l + dr.l + bs.l + ar.l + fx.l + 0.55 * wet_l
    right = music.r + dr.r + bs.r + ar.r + fx.r + 0.55 * wet_r
    mix = np.vstack([left, right])[:, :N]
    mix = highpass(mix, 28)

    # Fade with the picture at the end.
    t = np.arange(N) / SR
    fade = np.clip((179.6 - t) / 1.9, 0, 1) ** 1.5
    mix *= fade
    return mix


def loudness(mix):
    """Integrated loudness (LUFS) measured by ffmpeg's EBU R128 filter."""
    import os
    import re
    import subprocess
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        write_wav(path, mix / max(1.0, float(np.max(np.abs(mix)))))
        out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128",
                              "-f", "null", "-"], capture_output=True, text=True).stderr
    finally:
        os.remove(path)
    scale = max(1.0, float(np.max(np.abs(mix))))
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1]) + 20 * np.log10(scale)


def limit(mix, ceiling=0.84, look=0.005, release=0.12):
    """Look-ahead peak limiter: smooth gain reduction, no clipping."""
    from scipy.ndimage import maximum_filter1d
    peak = np.max(np.abs(mix), axis=0)
    need = np.maximum(peak / ceiling, 1.0)
    w = s_(look) * 2 + 1
    need = maximum_filter1d(need, w)
    # Instant attack, exponential release (on the reduction in dB).
    red = 20 * np.log10(need)
    a = np.exp(-1.0 / (release * SR))
    red = signal.lfilter([1 - a], [1, -a], red)
    red = np.maximum(red, 20 * np.log10(maximum_filter1d(need, w)))
    gain = 10 ** (-red / 20)
    gain = np.roll(gain, -s_(look))
    return mix * gain


def master(mix, target=-16.0):
    lufs = loudness(mix)
    mix = mix * 10 ** ((target - lufs) / 20)
    return limit(mix).astype(np.float32)


def write_wav(path, mix):
    from scipy.io import wavfile
    pcm = np.clip(mix.T * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(path, SR, pcm)


def build(path):
    from .film import events
    mix = master(render(events()))
    write_wav(path, mix)
    return mix
