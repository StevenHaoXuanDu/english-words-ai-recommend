"""Shot timing for the video: the one schedule picture and music both follow.

    python3 timeline.py   -> BUILD/timeline.json

The video has no voice-over; the script's narration appears as on-screen captions.
Shot lengths come from script.md, stretched when a shot's captions need more
reading time. All times in timeline.json are absolute seconds from the start.
"""
import json
import os

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
    dict(id="S01", ch=1, base=3.0, chunks=[]),
    dict(id="S02", ch=1, base=6.0, lead=0.8, tail=1.0,
         chunks=["一个小村子，住着一百个人。"]),
    dict(id="S03", ch=1, base=6.0, lead=0.4,
         chunks=["对同一件小事，每个人的看法，都有一点不同。"]),
    dict(id="S04", ch=1, base=7.0, lead=0.4, tail=1.2,
         chunks=["这个村子只有一条规则：", ("你只听，和你差不多的人。", False)]),
    dict(id="S05", ch=2, base=9.0, lead=0.6,
         chunks=["“差不多”，不是住得近，是想得近。", "它只听括号里的这些人。"]),
    dict(id="S06", ch=2, base=8.0, lead=0.4, tail=1.2,
         chunks=["每一轮，它看看这些人，把自己的看法，挪到他们的平均值上。"]),
    dict(id="S07", ch=2, base=7.0, lead=0.6, gap=0.3, tail=0.6,
         chunks=["所有人同时这么做。", "一轮，又一轮。", "没有别的规则。"]),
    dict(id="S08", ch=3, base=6.0, lead=0.4,
         chunks=["第一次实验：每个人愿意听的范围，宽一些。"]),
    dict(id="S09", ch=3, base=10.0, lead=0.4,
         chunks=["最先动的，是两头的人。", "然后，整排人一点点向中间收拢。"]),
    dict(id="S10", ch=3, base=6.0, lead=0.3, tail=1.7,
         chunks=["第四轮。", "一百个人，站到了一起。"]),
    dict(id="S11", ch=4, base=10.0, lead=0.4,
         chunks=["同样的人，同样的起点，同样的规则。", "只改一件事：把这个范围，收窄三分之一。"]),
    dict(id="S12", ch=4, base=9.0, lead=0.3, gap=0.25, tail=0.6,
         chunks=["人们仍然在靠近。", "只是，只靠近身边的人。", "中间的几个人，被两边各自拉走。"]),
    dict(id="S13", ch=4, base=5.0, lead=0.3, tail=1.2,
         chunks=["第六轮之后，再也没有任何变化。"]),
    dict(id="S14", ch=4, base=12.0, lead=0.6, gap=0.55, tail=3.2,
         chunks=["没有人撒谎。", "没有人被收买。", "没有人是坏人。", "大家只是，都在听和自己差不多的人。"]),
    dict(id="S15", ch=5, base=7.0, lead=0.8,
         chunks=["现实里没有这个旋钮。", "但有很多东西，在悄悄收窄括号。"]),
    dict(id="S16", ch=5, base=13.0, lead=0.3, tail=0.5, beats=3.0,
         chunks=["推荐算法，喂你爱看的。", "关注列表，是你亲手挑的。", "拉黑，让一种颜色消失。", "评论区，先问你站哪边。"]),
    dict(id="S17", ch=5, base=9.0, lead=0.4,
         chunks=["它们未必为分裂而设计。", "但它们都让我们，更容易只听见差不多的人。"]),
    dict(id="S18", ch=6, base=8.0, lead=0.6,
         chunks=["回到村子。", "这一次，有一个人，开始去听远处的颜色。"]),
    dict(id="S19", ch=6, base=8.0, lead=0.4,
         chunks=["它走到了中间。", "它听得见两边，两边也还听得见它。"]),
    dict(id="S20", ch=6, base=10.0, lead=0.4, gap=1.2, tail=1.8,
         chunks=["很久，几乎什么也没发生。", "然后，裂缝合上了。"]),
    dict(id="S21", ch=6, base=8.0, lead=0.4,
         chunks=["这并不总能成功。", "两边离得太远，就没有人能同时被两边听见。"]),
    dict(id="S22", ch=6, base=10.0, lead=1.0, tail=6.6,
         chunks=["你最近一次，认真听完一个反对意见，是什么时候？"]),
]

# Bottom annotations. A string covers the whole shot; a list of (text, start, end)
# uses shot-relative seconds, where end may be None (= shot end) and the special
# strings "c1s"/"c1e"/... name chunk starts/ends. "{r}" is a live round counter.
ANNOTATIONS = {
    "S01": "N = 1",
    "S02": "N = 100 · OPINION x ∈ [0, 1]",
    "S03": "INITIAL STATE · EVENLY SPACED · x = 0.00 … 1.00",
    "S04": "ONE RULE · BOUNDED CONFIDENCE",
    "S05": "NEIGHBORS(i) = { j : |xᵢ − xⱼ| ≤ ε }",
    "S06": "xᵢ(t+1) = mean{ xⱼ : |xᵢ − xⱼ| ≤ ε }",
    "S07": "SYNCHRONOUS UPDATE · ROUND 1 (DEMO, ε = 0.20)",
    "S08": "EXPERIMENT 1 · BOUNDED CONFIDENCE · ε = 0.30 · Hegselmann & Krause, 2002",
    "S09": "ROUND 1 → 3 · SPREAD 1.00 → 0.71 → 0.46 → 0.29",
    "S10": "ROUND 4 · CONSENSUS · 1 CLUSTER · x = 0.50",
    "S11": "EXPERIMENT 2 · ε = 0.30 → 0.20",
    "S12": "ROUND 3 → 6 · GROUPS 9 → 8 → 4 → 2",
    "S13": "ROUND {r} · 2 CLUSTERS · 50 | 50 · GAP 0.39 > ε",
    "S14": "NO VILLAINS · SAME PEOPLE · SAME RULE · NARROWER ε",
    "S15": "REAL WORLD · NO SINGLE ε",
    "S16": [("RECOMMENDATION FEED", "c1s", "c2s"), ("FOLLOW LIST", "c2s", "c3s"),
            ("BLOCK", "c3s", "c4s"), ("REPLY THREAD · PICK A SIDE", "c4s", None)],
    "S17": "ILLUSTRATIVE · A MODEL, NOT A MEASUREMENT",
    "S18": "1 AGENT · ε : 0.20 → 0.40 · EVERYONE ELSE UNCHANGED",
    "S19": "ROUND 1 · BRIDGE x = 0.50 · DISTANCE 0.19 ≤ ε, BOTH SIDES",
    "S20": [("ROUND {r} · BRIDGE ε = 0.40 · OTHERS ε = 0.20", 0.0, "merge1"),
            ("MERGED AT ROUND 36 · THE SPLIT TOOK 6", "merge1", None)],
    "S21": "REQUIRES GAP ≤ 2ε · HERE 0.39 ≤ 0.40",
    "S22": [("ε = ?", 0.6, "c1e+0.8"),
            ("Hegselmann & Krause, JASSS 5(3), 2002 · SIMULATION: N = 100, EVENLY SPACED START",
             "c1e+1.0", "c1e+3.6")],
}


def reading_time(text):
    """Seconds a caption stays up: unhurried reading, well under subtitle-speed limits."""
    return 0.6 + len(text) / 6.5


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


def shot_events(sid, d, c):
    """Shot-relative event windows [t0, t1]. c = list of (start, end) per chunk."""
    cs = lambda k: c[k - 1][0]
    ce = lambda k: c[k - 1][1]
    if sid == "S01":
        return {"rect_in": [0.3, 2.0]}
    if sid == "S02":
        return {"light_up": [0.3, 4.5]}
    if sid == "S03":
        return {"pan": [0.0, d]}
    if sid == "S04":
        return {"dim": [0.1, 0.7], "card": [cs(2) - 0.1, ce(2) + 1.0], "undim": [d - 0.8, d - 0.2]}
    if sid == "S05":
        return {"zoom_in": [0.0, 2.5], "bracket_in": [cs(2) - 0.6, cs(2) + 0.4]}
    if sid == "S06":
        return {"rays": [cs(1) + 0.6, cs(1) + 2.2], "move": [cs(1) + 2.4, ce(1) + 0.2]}
    if sid == "S07":
        return {"zoom_out": [0.0, 1.2], "reset_demo": [0.0, 1.0], "flash": [1.2, 2.2],
                "round:1": [2.3, 4.1], "rewind": [d - 1.6, d - 0.3]}
    if sid == "S08":
        return {"bracket_in": [0.3, 1.3], "bracket_out": [d - 1.0, d - 0.2]}
    if sid == "S09":
        return rounds(1, 3, 0.6, d - 0.3, frac=0.7)
    if sid == "S10":
        return {"round:4": [0.0, 1.6], "glow": [1.6, 2.6]}
    if sid == "S11":
        dur2 = ce(2) - cs(2)
        return {"rewind": [0.2, 2.6], "bracket_in": [2.6, 3.2],
                "narrow": [cs(2) + 0.35 * dur2, ce(2)]}
    if sid == "S12":
        ev = {"bracket_out": [0.0, 0.5]}
        ev.update(rounds(1, 6, 0.4, d - 0.3, frac=0.75))
        return ev
    if sid == "S13":
        return {"fast_forward": [0.8, d - 0.6]}
    if sid == "S14":
        ne = ce(len(c))
        return {"push": [0.0, d], "silence": [ne + 0.1, ne + 3.1], "piano": [ne + 3.1, ne + 3.1]}
    if sid == "S15":
        return {"reset": [0.0, 1.2], "bracket_in": [1.4, 2.0]}
    if sid == "S16":
        return {f"beat:{k}": [cs(k), cs(k) + 0.6] for k in range(1, 5)}
    if sid == "S17":
        ev = {"restore": [0.0, 0.6]}
        ev.update(rounds(1, 3, 0.8, 4.6, frac=0.8, accel=1.3))
        ev["freeze"] = [4.6, 4.6]
        return ev
    if sid == "S18":
        return {"reset": [0.0, 1.0], "highlight": [1.0, 2.0],
                "bracket_in": [cs(2), cs(2) + 0.5], "widen": [cs(2) + 0.5, ce(2)]}
    if sid == "S19":
        return {"round:1": [0.4, 2.4], "threads": [2.6, 3.6]}
    if sid == "S20":
        ev = rounds(2, 35, 0.3, cs(2) + 0.2, frac=0.8, accel=1.06)
        ev["round:36"] = [cs(2) + 0.5, cs(2) + 1.4]
        ev["merge"] = [cs(2) + 1.4, cs(2) + 1.4]
        return ev
    if sid == "S21":
        dur2 = ce(2) - cs(2)
        b = cs(2) + 0.55 * dur2
        return {"dim": [0.2, 0.8], "diagram_in": [cs(2) - 0.2, cs(2) + 0.4],
                "break": [b, b + 0.3], "diagram_out": [d - 1.4, d - 0.6], "undim": [d - 1.0, d - 0.2]}
    if sid == "S22":
        e = ce(1)
        return {"collapse": [0.0, 1.4], "card": [cs(1) - 0.2, e + 3.6], "black": [e + 3.6, e + 4.2]}
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
    else:
        raise ValueError(spec)
    return t + (float(off) if off else 0.0)


def build():
    t = 0.0
    shots = []
    for s in SHOTS:
        lead, gap, tail = s.get("lead", 0.5), s.get("gap", 0.4), s.get("tail", 0.8)
        items = [(x, True) if isinstance(x, str) else x for x in s["chunks"]]
        lens = [reading_time(txt) for txt, _ in items]
        c, cur = [], lead
        for k, L in enumerate(lens):
            if "beats" in s:
                cur = lead + sum(max(s["beats"], lens[j] + 0.6) for j in range(k))
            c.append((cur, cur + L))
            cur += L + gap
        need = (c[-1][1] + tail) if c else 0.0
        if "beats" in s:
            need = lead + sum(max(s["beats"], L + 0.6) for L in lens) + tail
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
                    entry["counter"] = {"from": 6, "to": 100,
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
