"""Shot timing for the video: the one schedule picture and music both follow.

    python3 timeline.py   -> BUILD/timeline.json

The video has no voice-over; the script's narration appears as on-screen captions.
Shot lengths come from script.md, stretched when a shot's captions need more
reading time. All times in timeline.json are absolute seconds from the start.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get(
    "BUILD",
    "/tmp/claude-0/-home-user-english-words-ai-recommend/"
    "4af3582d-b5b8-57c7-bf82-51aca3b2d552/scratchpad/build",
)
FPS = 30

# Each chunk is one caption. A chunk given as (text, False) gets no caption
# because a title card shows the same words.
SHOTS = [
    dict(id="S01", ch=1, base=1.8, chunks=[]),
    dict(id="S02", ch=1, base=3.7, lead=0.5, tail=0.8,
         chunks=["一个小村子，住着一百个人。"]),
    dict(id="S03", ch=1, base=4.5, lead=0.25, tail=0.8,
         chunks=["对同一件小事，每个人的看法，都有一点不同。"]),
    # S04: the title card (chunk 2) starts while caption 1 is still up (negative gap: allowed
    # before a chunk that is shown as a card, not as a caption)
    dict(id="S04", ch=1, base=4.2, lead=0.2, gap=-0.55, tail=1.1,
         chunks=["这个村子只有一条规则：", ("你只听，和你差不多的人。", False)]),
    dict(id="S05", ch=2, base=5.0, lead=0.35, gap=0.35, tail=0.7,
         chunks=["“差不多”，不是住得近，是想得近。", "它只听括号里的这些人。"]),
    dict(id="S06", ch=2, base=4.5, lead=0.25, tail=0.7,
         chunks=["每一轮，它看看这些人，把自己的看法，挪到他们的平均值上。"]),
    dict(id="S07", ch=2, base=5.0, lead=0.35, gap=0.4, tail=0.45,
         chunks=["所有人同时这么做。", "一轮，又一轮。", "没有别的规则。"]),
    dict(id="S08", ch=3, base=4.0, lead=0.3, tail=0.55,
         chunks=["第一次实验：每个人愿意听的范围，宽一些。"]),
    dict(id="S09", ch=3, base=5.0, lead=0.3, gap=0.6, tail=0.5,
         chunks=["最先动的，是两头的人。", "然后，整排人一点点向中间收拢。"]),
    dict(id="S10", ch=3, base=4.0, lead=0.2, gap=0.35, tail=0.45,
         chunks=["第四轮。", "一百个人，站到了一起。"]),
    dict(id="S11", ch=4, base=6.0, lead=0.35, gap=0.45, tail=0.5,
         chunks=["同样的人，同样的起点，同样的规则。", "只改一件事：把这个范围，收窄三分之一。"]),
    dict(id="S12", ch=4, base=6.0, lead=0.25, gap=0.35, tail=0.5,
         chunks=["人们仍然在靠近。", "只是，只靠近身边的人。", "中间的几个人，被两边各自拉走。"]),
    dict(id="S13", ch=4, base=3.0, lead=0.25, tail=0.5,
         chunks=["第七轮之后，再也没有任何变化。"]),
    dict(id="S14", ch=4, base=9.0, lead=0.4, gap=0.55, tail=1.8,
         chunks=["没有人撒谎。", "没有人被收买。", "没有人是坏人。", "大家只是，都在听和自己差不多的人。"]),
    dict(id="S15", ch=5, base=5.0, lead=0.7, gap=0.4, tail=0.5,
         chunks=["现实里没有这个旋钮。", "但有很多东西，在悄悄收窄括号。"]),
    dict(id="S16", ch=5, base=9.0, lead=0.3, tail=0.4, beats=2.6,
         chunks=["推荐算法，喂你爱看的。", "关注列表，是你亲手挑的。", "拉黑，让一种颜色消失。", "评论区，先问你站哪边。"]),
    dict(id="S17", ch=5, base=5.0, lead=0.3, gap=0.35, tail=0.8,
         chunks=["它们未必为分裂而设计。", "但它们都让我们，更容易只听见差不多的人。"]),
    dict(id="S18", ch=6, base=5.0, lead=0.6, gap=0.4, tail=0.5,
         chunks=["回到村子。", "这一次，有一个人，开始去听远处的颜色。"]),
    dict(id="S19", ch=6, base=4.5, lead=0.3, gap=0.35, tail=0.7,
         chunks=["它走到了中间。", "它听得见两边，两边也还听得见它。"]),
    dict(id="S20", ch=6, base=5.0, lead=0.3, gap=0.85, tail=1.5,
         chunks=["很久，几乎什么也没发生。", "然后，裂缝合上了。"]),
    dict(id="S21", ch=6, base=5.0, lead=0.3, gap=0.4, tail=0.8,
         chunks=["这并不总能成功。", "两边离得太远，就没有人能同时被两边听见。"]),
    dict(id="S22", ch=6, base=7.0, lead=0.9, tail=4.4,
         chunks=[("你最近一次，认真听完一个反对意见，是什么时候？", False)]),
]

# Bottom annotations. A string covers the whole shot; a list of (text, start, end)
# uses shot-relative seconds, where end may be None (= shot end) and the special
# strings "c1s"/"c1e"/... name chunk starts/ends. "{r}" is a live round counter.
ANNOTATIONS = {
    "S01": [("N = 1", "rect_in+0.35", None)],
    "S02": "N = 100 · OPINION x ∈ [0, 1]",
    "S03": "INITIAL STATE · EVENLY SPACED · x = 0.00 … 1.00",
    "S04": "ONE RULE · BOUNDED CONFIDENCE",
    "S05": "NEIGHBORS(i) = { j : |xᵢ − xⱼ| ≤ ε }",
    "S06": "xᵢ(t+1) = mean{ xⱼ : |xᵢ − xⱼ| ≤ ε }",
    "S07": [("SYNCHRONOUS UPDATE · ROUND {r} (DEMO, ε = 0.20)", "c1s", "rewind+0.15")],
    "S08": "EXPERIMENT 1 · BOUNDED CONFIDENCE · ε = 0.30 · Hegselmann & Krause, 2002",
    "S09": "ROUND 1 → 3 · SPREAD 1.00 → 0.71 → 0.46 → 0.29",
    "S10": "ROUND 4 · CONSENSUS · 1 CLUSTER · x = 0.50",
    "S11": "EXPERIMENT 2 · ε = 0.30 → 0.20",
    "S12": [("ROUND 3 → 7 · GROUPS 9 → 8 → 4 → 2 → 2", "round:3", None)],
    "S13": "ROUND {r} · 2 CLUSTERS · 50 | 50 · GAP 0.39 > ε",
    "S14": "NO VILLAINS · SAME PEOPLE · SAME RULE · NARROWER ε",
    "S15": "REAL WORLD · NO SINGLE ε",
    "S16": [("RECOMMENDATION FEED", "c1s", "c2s"), ("FOLLOW LIST", "c2s", "c3s"),
            ("BLOCK", "c3s", "c4s"), ("REPLY THREAD · PICK A SIDE", "c4s", None)],
    "S17": "ILLUSTRATIVE · A MODEL, NOT A MEASUREMENT",
    "S18": "1 AGENT · ε : 0.20 → 0.40 · EVERYONE ELSE UNCHANGED",
    "S19": [("ROUND 1 · BRIDGE x = 0.50 · DISTANCE 0.19 ≤ ε, BOTH SIDES", "round:1+0.3", None)],
    "S20": [("ROUND {r} · BRIDGE ε = 0.40 · OTHERS ε = 0.20", 0.0, "merge"),
            ("MERGED AT ROUND 36 · THE SPLIT SETTLED AT ROUND 7", "merge", None)],
    "S21": "REQUIRES GAP ≤ 2ε · HERE 0.39 ≤ 0.40",
    "S22": [("ε = ?", 0.5, "c1e+0.2"),
            ("Hegselmann & Krause, JASSS 5(3), 2002 · SIMULATION: N = 100, EVENLY SPACED START",
             "c1e+0.2", "c1e+4.1")],
}


READ_MIN = 1.2          # s: no caption is up for less than this
MIN_GAP = 0.6           # s between consecutive captions of a shot: room for the exit / reveal
MERGE_CONTACT = 0.5115  # fraction of the S20 round-36 tween at which the blocks touch


def reading_time(text):
    """Seconds a caption is scheduled for: 0.45 s + 7.5 characters per second, at least
    READ_MIN. The renderer shows each caption 0.2 s before and up to 0.35 s after its slot
    (MIN_GAP keeps room for that), so the time a caption is fully revealed and not yet leaving
    stays above 0.45 s + len / 8 (<= 8 characters per second) and above READ_MIN."""
    return max(READ_MIN, 0.45 + len(text) / 7.5)


def rounds(first, last, t0, t1, frac=0.75, accel=1.0):
    """Spread round tweens first..last over [t0, t1]. accel > 1 makes later rounds faster."""
    n = last - first + 1
    w = [accel ** -k for k in range(n)]
    total = sum(w)
    ev, t = {}, t0
    for k, r in enumerate(range(first, last + 1)):
        slot = (t1 - t0) * w[k] / total
        ev[f"round:{r}"] = [t, t + slot * frac]
        t += slot
    return ev


CREEP_RAMP = (0.22, 0.14)     # S20 creep: speed ramps up over this fraction, eases out over this


def creep_profile(u, ramp_in=CREEP_RAMP[0], ramp_out=CREEP_RAMP[1]):
    """Fraction (0..1) of a continuous creep done at progress u (0..1, scalar or array): its
    speed ramps up (smoothstep) over the first ramp_in, holds, and eases to rest over the last
    ramp_out. render_shots evaluates the very same profile for S20's fractional round index."""
    m = 4000
    g = np.linspace(0.0, 1.0, m + 1)
    a = np.clip(g / ramp_in, 0, 1)
    b = np.clip((1 - g) / ramp_out, 0, 1)
    v = a * a * (3 - 2 * a) * (b * b * (3 - 2 * b))
    f = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))])
    return np.interp(u, g, f / f[-1])


def creep(first, last, t0, t1):
    """Contiguous round windows for a continuous creep through rounds first..last over [t0, t1]:
    round k's window is where the profile's fractional round index runs from k-1 to k. The
    renderer interpolates the states along that one index (no per-round easing, no hold), so
    every state is passed through exactly and the motion never stutters."""
    n = last - first + 1
    g = np.linspace(0.0, 1.0, 20001)
    edges = np.interp(np.arange(n + 1) / n, creep_profile(g), g)
    return {f"round:{first + k}": [t0 + (t1 - t0) * float(edges[k]), t0 + (t1 - t0) * float(edges[k + 1])]
            for k in range(n)}


def shot_events(sid, d, c):
    """Shot-relative event windows [t0, t1]. c = list of (start, end) per chunk."""
    cs = lambda k: c[k - 1][0]
    ce = lambda k: c[k - 1][1]
    if sid == "S01":
        return {"rect_in": [0.2, 1.05]}
    if sid == "S02":
        return {"light_up": [0.1, 2.4]}
    if sid == "S03":
        return {"pan": [0.0, d]}
    if sid == "S04":
        return {"dim": [0.05, 0.5], "card": [cs(2) - 0.2, cs(2) + 3.05], "undim": [d - 0.75, d - 0.1]}
    if sid == "S05":
        return {"zoom_in": [0.0, 1.2], "bracket_in": [cs(2) - 0.5, cs(2) + 0.35]}
    if sid == "S06":
        return {"rays": [cs(1) + 0.35, cs(1) + 1.45], "move": [cs(1) + 1.5, cs(1) + 3.0]}
    if sid == "S07":
        return {"reset_demo": [0.0, 0.55], "zoom_out": [0.45, 1.3], "flash": [1.25, 1.7],
                "round:1": [1.7, 2.5], "round:2": [cs(2) + 0.45, cs(2) + 1.25],
                "rewind": [d - 1.3, d - 0.15]}
    if sid == "S08":
        return {"sweep": [0.0, 0.75], "bracket_in": [0.2, 0.9], "bracket_out": [d - 0.55, d - 0.1]}
    if sid == "S09":
        return rounds(1, 3, 0.3, d - 0.15, frac=0.72)
    if sid == "S10":
        return {"round:4": [0.0, 0.85], "glow": [0.85, 1.7], "accent": [0.85, 2.3]}
    if sid == "S11":
        dur2 = ce(2) - cs(2)
        return {"rewind": [0.05, 1.75], "bracket_in": [1.8, 2.3],
                "narrow": [cs(2) + 0.35 * dur2, ce(2)]}
    if sid == "S12":
        ev = {"bracket_out": [0.0, 0.35]}
        ev.update(rounds(1, 7, 0.3, d - 0.2, frac=0.7))
        return ev
    if sid == "S13":
        return {"lock": [0.0, 1.2], "measure": [0.15, d - 0.25], "fast_forward": [0.5, d - 0.35]}
    if sid == "S14":
        ne = ce(len(c))
        return {"push": [0.0, d], "silence": [ne + 0.1, ne + 1.5], "piano": [ne + 1.5, ne + 1.5]}
    if sid == "S15":
        # push-through transition: the split zooms past the camera, the spectrum lands
        return {"reset": [0.0, 0.95], "bracket_in": [1.05, 1.55]}
    if sid == "S16":
        return {f"beat:{k}": [cs(k), cs(k) + 0.6] for k in range(1, 5)}
    if sid == "S17":
        ev = {"restore": [0.0, 0.5]}
        ev.update(rounds(1, 3, 0.6, 3.3, frac=0.8, accel=1.3))
        ev["freeze"] = [3.3, 3.3]
        return ev
    if sid == "S18":
        # light-sweep wipe from S17's frozen fracture to the split, then the bridge lights up
        return {"reset": [0.0, 0.95], "highlight": [0.95, 1.6],
                "bracket_in": [cs(2), cs(2) + 0.4], "widen": [cs(2) + 0.4, ce(2) - 0.2]}
    if sid == "S19":
        return {"round:1": [0.25, 1.25], "threads": [1.4, 2.2]}
    if sid == "S20":
        ev = creep(2, 35, 0.1, cs(2))
        ev["creep"] = [0.1, cs(2)]
        ev["round:36"] = [cs(2) + 0.2, cs(2) + 0.95]
        # the blocks touch at MERGE_CONTACT of the round-36 tween (render_shots computes it from
        # the tween and checks this value)
        ev["merge"] = [cs(2) + 0.2 + MERGE_CONTACT * 0.75] * 2
        return ev
    if sid == "S21":
        # dim: the real block shrinks, sinks and dims (stage for the inset); the inset shows the
        # real bridge-round-1 geometry, opens each block's listening bracket, then (hypothetically)
        # the blocks drift apart: strain until the brackets stop overlapping at the bridge, break,
        # release; undim brings the real block back for S22
        dur2 = ce(2) - cs(2)
        b = cs(2) + 0.42 * dur2
        return {"dim": [0.0, 0.75], "diagram_in": [cs(1) + 0.25, cs(1) + 0.8],
                "bracket_l": [cs(1) + 0.85, cs(1) + 1.35], "bracket_r": [cs(1) + 1.15, cs(1) + 1.65],
                "strain": [b - 1.1, b], "break": [b, b + 0.3], "release": [b, b + 1.25],
                "diagram_out": [d - 0.95, d - 0.4], "undim": [d - 0.8, d - 0.05]}
    if sid == "S22":
        e = ce(1)
        # the question holds to the end; the background recedes first, then card, citation and
        # the last rect fade together (about 0.15 s of true black remain)
        return {"collapse": [0.0, 1.1], "card": [cs(1) - 0.2, e + 3.95], "black": [e + 3.5, e + 4.1],
                "bg_out": [e + 3.0, e + 4.25]}
    return {}


def resolve(spec, c, ev, d):
    """Turn an annotation time spec into shot-relative seconds."""
    if spec is None:
        return d
    if isinstance(spec, (int, float)):
        return float(spec)
    base, _, off = spec.partition("+")
    if base.startswith("c") and base[-1] in "se":
        k = int(base[1:-1])
        t = c[k - 1][0] if base[-1] == "s" else c[k - 1][1]
    elif base == "merge1":
        t = ev["round:36"][1]
    elif base in ev:                      # an event name: its start
        t = ev[base][0]
    else:
        raise ValueError(spec)
    return t + (float(off) if off else 0.0)


def build():
    t = 0.0
    shots = []
    for s in SHOTS:
        lead, tail = s.get("lead", 0.5), s.get("tail", 0.8)
        items = [(x, True) if isinstance(x, str) else x for x in s["chunks"]]
        lens = [reading_time(txt) for txt, _ in items]
        c, cur = [], lead
        for k, L in enumerate(lens):
            if "beats" in s:
                cur = lead + sum(max(s["beats"], lens[j] + 0.5) for j in range(k))
            elif k:                       # gap before chunk k; a card chunk may overlap
                g = s.get("gap", 0.4)
                cur = c[-1][1] + (g if not items[k][1] else max(MIN_GAP, g))
            c.append((cur, cur + L))
            cur += L
        need = (c[-1][1] + tail) if c else 0.0
        if "beats" in s:
            need = lead + sum(max(s["beats"], L + 0.5) for L in lens) + tail
        d = round(max(s["base"], need) * FPS) / FPS
        ev = shot_events(s["id"], d, c)

        ann_spec = ANNOTATIONS.get(s["id"])
        if isinstance(ann_spec, str):
            ann_spec = [(ann_spec, 0.0, None)]
        anns = []
        for text, a, b in ann_spec or []:
            a, b = resolve(a, c, ev, d), resolve(b, c, ev, d)
            entry = {"text": text, "start": t + a, "end": t + b}
            if "{r}" in text:
                if s["id"] == "S13":
                    entry["counter"] = {"from": 7, "to": 100,
                                        "t0": t + ev["fast_forward"][0], "t1": t + ev["fast_forward"][1]}
                else:  # S20 follows the round tweens; renderer shows the round being entered
                    entry["counter"] = {"rounds": "events"}
            anns.append(entry)

        shots.append({
            "id": s["id"], "chapter": s["ch"], "start": t, "end": t + d,
            "captions": [{"id": f"{s['id']}_{k}", "text": txt, "show": sub,
                          "start": t + a, "end": t + b}
                         for k, ((txt, sub), (a, b)) in enumerate(zip(items, c))],
            "annotations": anns,
            "events": {k: [t + v[0], t + v[1]] for k, v in ev.items()},
        })
        t += d
    return {"fps": FPS, "width": 1920, "height": 1080, "duration": t, "shots": shots}


if __name__ == "__main__":
    os.makedirs(BUILD, exist_ok=True)
    tl = build()
    with open(os.path.join(BUILD, "timeline.json"), "w") as f:
        json.dump(tl, f, ensure_ascii=False, indent=1)
    m, s = divmod(tl["duration"], 60)
    print(f"total {int(m)}:{s:05.2f} · {len(tl['shots'])} shots -> {BUILD}/timeline.json")
    for sh in tl["shots"]:
        print(f"  {sh['id']} {sh['start']:7.2f}–{sh['end']:7.2f} ({sh['end'] - sh['start']:.2f}s)")
