# pro-english-words

A Claude skill that turns a plain English or Chinese word into a list of more professional English alternatives.

一个 Claude Skill：输入一个英文或中文词，直接返回 10 个更专业的英文表达。

## Examples

**Input:** `significant`

```
- substantial
- considerable
- notable
- pivotal
- profound
- consequential
- paramount
- momentous
- noteworthy
- instrumental
```

**Input:** `理解`

```
- comprehend — 理解
- grasp — 掌握要点
- appreciate — 体会、认同
- recognize — 认识到
...
```

## Features

- English or Chinese input
- Word list only, no explanations (ask for "例句" / "explain" if you want them)
- Same part of speech as the input, common words first
- Silent typo correction
- Multiple words → grouped lists

## Install

1. Download this folder as a ZIP (or zip the `pro-english-words` folder).
2. In Claude, go to **Settings → Capabilities → Skills** and upload the ZIP.

## License

MIT
