---
name: pro-english-words
description: Use when the user sends a single English or Chinese word/phrase and wants more professional, advanced English alternatives as a bare list.
---

# Pro English Words

The user sends a word or short phrase (English or Chinese) and wants more professional, polished English alternatives. Output only the list. No intro, no explanations, no example sentences, no closing line.

## English input

Return 10 more professional / advanced English alternatives as a bullet list.

```
- substantial
- considerable
- notable
...
```

## Chinese input

Translate to English, then return 10 professional English options. Put a very short Chinese gloss (2–6 characters) after each word so the user can pick the right sense.

Input: 理解
```
- comprehend — 理解
- grasp — 掌握要点
- appreciate — 体会、认同
- recognize — 认识到
...
```

If the Chinese has several distinct meanings, cover the main ones in the same list (most common sense first).

## Rules

- Keep the same part of speech as the input (noun → nouns, adverb → adverbs). For phrases, give phrases.
- Order: most common and safe-to-use in professional writing first, rarer/formal ones last.
- Prefer words a native speaker would actually use at work or in academic writing; avoid obscure or archaic words.
- Silently correct typos (e.g. "intellligence" → intelligence). Do not mention the correction.
- Multiple words in one message: give a separate list for each, with the input word as a bold label above its list.
- Never repeat the input word itself in the list.
- Only add explanations if the user explicitly asks (e.g. "explain", "例句", "区别").
