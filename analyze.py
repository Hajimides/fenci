#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《骆驼祥子》文本计量分析。

用法（在项目根目录）：
    .venv/bin/python analyze.py
    .venv/bin/python analyze.py --input 骆驼祥子.txt --outdir output

产出：
    output/word_freq.txt
    output/character_keywords.txt
    output/emotion_analysis.txt
    index.html（GitHub Pages 首页）
"""

from __future__ import annotations

import argparse
import html
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import jieba.posseg as pseg
import pandas as pd
from cnsenti import Emotion, Sentiment
from opencc import OpenCC

ROOT = Path(__file__).resolve().parent
LEXICON_DIR = ROOT / "lexicons"

CHAPTER_NUMS = [
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
    "二十一", "二十二", "二十三", "二十四",
]
CHAPTER_SET = set(CHAPTER_NUMS)

CHARACTERS = {
    "祥子": ["祥子"],
    "虎妞": ["虎妞", "虎姑娘"],
    "刘四爷": ["刘四爷", "刘老头子", "老头子", "刘四"],
    "小福子": ["小福子"],
    "曹先生": ["曹先生"],
    "高妈": ["高妈"],
    "二强子": ["二强子"],
    "老程": ["老程"],
    "阮明": ["阮明"],
    "夏太太": ["夏太太"],
    "杨太太": ["杨太太"],
    "孙侦探": ["孙侦探"],
    "小马儿": ["小马儿"],
}

NAME_TOKENS = {alias for aliases in CHARACTERS.values() for alias in aliases}
ALIAS_TO_CANON = {alias: canon for canon, aliases in CHARACTERS.items() for alias in aliases}

# 单字内容词：本书主题极强，去停用词后仍应保留
UNIGRAM_KEEP = set(
    "车钱命苦汗恨哭血饿病死赌嫖骗懒穷雪雨风狗树热冷泪火冰酒烟刀枪"
    "爱恨善恶穷富强弱新旧香臭脏净美丑黑白"
)

CONTENT_POS = tuple("n v a i l z t s b j g".split())

# 方向补语、轻动词：高频但几乎没有主题信息
LIGHT_WORDS = {
    "起来", "出来", "出去", "回来", "过来", "过去", "进去", "进来",
    "上来", "下去", "上去", "下来", "看着", "看看", "告诉", "坐在",
    "放在", "找到", "跟着", "剩下", "露出", "说话", "完全", "有时候",
    "容易", "平日", "今天", "明天", "不到", "不知", "不想", "不会",
    "先生", "太太",
}

SENTI_UNIGRAM_POS = set("喜乐爱福暖香甜")
SENTI_UNIGRAM_NEG = set("苦恨哭怕饿病死脏懒赌嫖骗穷泪臭惨悲哀痛伤吓慌抢骂怒")
SENTI_BLACKLIST = {
    "是", "上", "下", "到", "想", "要", "能", "和", "过", "把", "说", "会", "便",
    "中", "对", "可", "好", "来", "去", "看", "有", "这", "那", "就", "也", "都",
    "还", "又", "很", "让", "给", "被", "从", "为", "以", "而", "但", "自己",
    "什么", "怎么", "怎样", "一个", "东西", "似乎", "老", "知道", "打", "人",
    "大", "小", "多", "少", "高", "低", "长", "短", "没", "不", "无", "非",
    "着", "了", "的", "地", "得", "在", "与", "及", "或", "比", "向", "往",
    "一定", "点头", "经验", "便宜", "不错", "热闹", "活着", "不肯", "厉害",
    "不大", "没法", "一气", "关系", "办法", "收拾", "坐下", "来到", "事儿",
    "一口", "显著", "不出", "是不是", "机会", "干脆", "根本", "没想到", "不便",
    "得到", "故意", "仆人",
}

XIANGZI_STAGES = [
    ("前期（第1–8章）买车与被夺", range(1, 9)),
    ("中期（第9–16章）虎妞与宅门", range(9, 17)),
    ("后期（第17–24章）崩溃与堕落", range(17, 25)),
]

PLOT_MARKERS = [
    (1, "第一次买车"),
    (2, "被大兵抢车"),
    (6, "与虎妞纠葛"),
    (11, "孙侦探敲诈"),
    (13, "回到人和厂"),
    (14, "刘四寿宴决裂"),
    (17, "小福子出现"),
    (23, "小福子之死"),
    (24, "出卖阮明"),
]

EMOTION_KEYS = ["好", "乐", "哀", "怒", "惧", "恶", "惊"]
EMOTION_ATTR = {
    "好": "Haos",
    "乐": "Les",
    "哀": "Ais",
    "怒": "Nus",
    "惧": "Jus",
    "恶": "Wus",
    "惊": "Jings",
}
EMOTION_COLOR = {
    "好": "#3d6b4f",
    "乐": "#c4a35a",
    "哀": "#5b7c99",
    "怒": "#a33b3b",
    "惧": "#6b5b95",
    "恶": "#5c5346",
    "惊": "#d07c3c",
}


def read_word_list(path: Path) -> list[str]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            words.append(w)
    return words


def load_stopwords() -> set[str]:
    return set(read_word_list(LEXICON_DIR / "stopwords.txt")) | LIGHT_WORDS


def normalize_text(raw: str) -> str:
    text = raw.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    cc = OpenCC("t2s")
    text = cc.convert(text)
    protect = ["著名", "显著", "著作", "执著", "昭著", "土著", "原著", "著者"]
    holders = {}
    for i, word in enumerate(protect):
        token = f"\x00P{i}\x00"
        holders[token] = word
        text = text.replace(word, token)
    text = (
        text.replace("著", "着")
        .replace("伕", "夫")
        .replace("舖", "铺")
        .replace("牠", "它")
    )
    for token, word in holders.items():
        text = text.replace(token, word)
    return text


def split_chapters(text: str) -> list[tuple[int, str]]:
    start = text.find("我们所要介绍的是祥子")
    if start < 0:
        start = text.find("我们所要介绍的是祥子")
    if start >= 0:
        text = text[start:]
    text = text.replace("（全书完）", "").replace("(全书完)", "")

    chapters: list[tuple[int, str]] = []
    current = 1
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in CHAPTER_SET:
            n = CHAPTER_NUMS.index(stripped) + 1
            if not buf:
                current = n
                continue
            chapters.append((current, "\n".join(buf).strip()))
            buf = []
            current = n
            continue
        if stripped.startswith("目录") or stripped.startswith("《骆驼祥子》"):
            continue
        buf.append(line)
    if buf:
        chapters.append((current, "\n".join(buf).strip()))

    merged: dict[int, list[str]] = defaultdict(list)
    for num, body in chapters:
        if body:
            merged[num].append(body)
    return [(n, "\n".join(merged[n])) for n in sorted(merged)]


def is_cjk(word: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in word)


def is_punct(word: str) -> bool:
    if not word:
        return True
    return all(unicodedata.category(ch).startswith("P") or ch.isspace() for ch in word)


def is_content(word: str, flag: str, stopwords: set[str]) -> bool:
    if not word or word in stopwords or not is_cjk(word) or is_punct(word):
        return False
    if word.isdigit():
        return False
    if len(word) == 1 and word not in UNIGRAM_KEEP:
        return False
    return any(flag.startswith(p) for p in CONTENT_POS)


def tokenize(text: str, stopwords: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (all_tokens, content_tokens, content_without_names)."""
    all_tokens: list[str] = []
    content: list[str] = []
    for pair in pseg.cut(text):
        word = pair.word.strip()
        flag = pair.flag or "x"
        if not word or is_punct(word) or flag == "x":
            continue
        all_tokens.append(word)
        if is_content(word, flag, stopwords):
            content.append(word)
    content_no_names = [w for w in content if w not in NAME_TOKENS]
    return all_tokens, content, content_no_names


def log_likelihood(o1: float, n1: float, o2: float, n2: float) -> float:
    if o1 + o2 == 0 or n1 <= 0 or n2 <= 0:
        return 0.0
    e1 = n1 * (o1 + o2) / (n1 + n2)
    e2 = n2 * (o1 + o2) / (n1 + n2)
    ll = 0.0
    if o1 > 0 and e1 > 0:
        ll += o1 * math.log(o1 / e1)
    if o2 > 0 and e2 > 0:
        ll += o2 * math.log(o2 / e2)
    return 2.0 * ll


def tfidf_top(chapter_docs: list[list[str]], k: int = 12) -> list[list[tuple[str, float, int]]]:
    n_docs = len(chapter_docs)
    df = Counter()
    for toks in chapter_docs:
        df.update(set(toks))
    results = []
    for toks in chapter_docs:
        tf = Counter(toks)
        n = len(toks) or 1
        scored = []
        for w, c in tf.items():
            idf = math.log((n_docs + 1) / (df[w] + 1)) + 1.0
            scored.append((w, (c / n) * idf, c))
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        results.append(scored[:k])
    return results


def bigrams(tokens: list[str], n: int = 30) -> list[tuple[str, int]]:
    pairs = [f"{a}{b}" for a, b in zip(tokens, tokens[1:]) if a != b]
    return Counter(pairs).most_common(n)


def character_keyness(
    all_tokens: list[str],
    content_set_fn,
    window: int = 8,
    min_count: int = 3,
    topn: int = 15,
) -> dict[str, list[tuple[str, int, float]]]:
    """Distinctive content words in a ±window around each character name."""
    mentions = defaultdict(list)
    for i, tok in enumerate(all_tokens):
        canon = ALIAS_TO_CANON.get(tok)
        if canon:
            mentions[canon].append(i)

    global_content = Counter(w for w in all_tokens if content_set_fn(w))
    n_global = sum(global_content.values()) or 1
    out: dict[str, list[tuple[str, int, float]]] = {}

    for canon, idxs in mentions.items():
        ctx = Counter()
        aliases = set(CHARACTERS[canon])
        for i in idxs:
            lo, hi = max(0, i - window), min(len(all_tokens), i + window + 1)
            for w in all_tokens[lo:hi]:
                if w in aliases:
                    continue
                if content_set_fn(w):
                    ctx[w] += 1
        n_ctx = sum(ctx.values()) or 1
        if canon == "祥子":
            floor = 6
        elif len(idxs) < 25:
            floor = 2
        else:
            floor = min_count
        ranked = []
        for w, c in ctx.items():
            if c < floor:
                continue
            rest = global_content[w] - c
            n_rest = max(n_global - n_ctx, 1)
            if (c / n_ctx) <= (rest / n_rest):
                continue
            ll = log_likelihood(c, n_ctx, rest, n_rest)
            ranked.append((w, c, ll))
        ranked.sort(key=lambda x: (-x[2], -x[1], x[0]))
        out[canon] = ranked[:topn]
    return out


def stage_keyness(
    chapter_tokens: list[list[str]],
    content_set_fn,
    min_count: int = 4,
    topn: int = 12,
) -> list[tuple[str, list[tuple[str, int, float]]]]:
    bags = []
    for label, chs in XIANGZI_STAGES:
        toks = []
        for n in chs:
            if 1 <= n <= len(chapter_tokens):
                toks.extend(w for w in chapter_tokens[n - 1] if content_set_fn(w))
        bags.append((label, Counter(toks)))

    results = []
    for i, (label, ctr) in enumerate(bags):
        others = Counter()
        for j, (_, c2) in enumerate(bags):
            if i != j:
                others.update(c2)
        n1 = sum(ctr.values()) or 1
        n2 = sum(others.values()) or 1
        ranked = []
        for w, c in ctr.items():
            if c < min_count:
                continue
            rest = others[w]
            if (c / n1) <= (rest / n2):
                continue
            ll = log_likelihood(c, n1, rest, n2)
            ranked.append((w, c, ll))
        ranked.sort(key=lambda x: (-x[2], -x[1], x[0]))
        results.append((label, ranked[:topn]))
    return results


def _usable_senti(word: str, unigrams: set[str], stopwords: set[str]) -> bool:
    if not word or word in stopwords or word in SENTI_BLACKLIST or word in LIGHT_WORDS:
        return False
    if len(word) == 1:
        return word in unigrams
    return is_cjk(word)


def build_sentiment_dicts(stopwords: set[str]) -> tuple[set[str], set[str], set[str], dict[str, set[str]]]:
    senti = Sentiment()
    emo = Emotion()
    poss = {w.strip() for w in senti.Poss if w and str(w).strip()}
    negs = {w.strip() for w in senti.Negs if w and str(w).strip()}
    both = poss & negs
    poss -= both
    negs -= both

    extra_pos = set(read_word_list(LEXICON_DIR / "sentiment_extra_pos.txt"))
    extra_neg = set(read_word_list(LEXICON_DIR / "sentiment_extra_neg.txt"))
    poss = (poss | extra_pos | SENTI_UNIGRAM_POS) - extra_neg
    negs = (negs | extra_neg | SENTI_UNIGRAM_NEG) - extra_pos

    denys = {str(w).strip() for w in senti.Denys if w and str(w).strip()}
    poss -= denys
    negs -= denys
    poss = {w for w in poss if _usable_senti(w, SENTI_UNIGRAM_POS, stopwords)}
    negs = {w for w in negs if _usable_senti(w, SENTI_UNIGRAM_NEG, stopwords)}

    buckets = {k: set() for k in EMOTION_KEYS}
    for key, attr in EMOTION_ATTR.items():
        buckets[key] = {str(w).strip() for w in getattr(emo, attr) if w and str(w).strip()}
    extra_path = LEXICON_DIR / "emotion_extra.txt"
    for line in extra_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        word, cat = line.split("\t", 1)
        word, cat = word.strip(), cat.strip()
        if cat not in buckets:
            continue
        for k in buckets:
            buckets[k].discard(word)
        buckets[cat].add(word)
    for key in buckets:
        uni = SENTI_UNIGRAM_POS if key in {"好", "乐"} else SENTI_UNIGRAM_NEG
        buckets[key] = {
            w for w in buckets[key]
            if _usable_senti(w, uni, stopwords) and w not in denys
        }
    return poss, negs, denys, buckets


def polarity_counts(
    tokens: list[str], poss: set[str], negs: set[str], denys: set[str]
) -> tuple[int, int, list[str], list[str]]:
    pos = neg = 0
    pos_hits: list[str] = []
    neg_hits: list[str] = []
    for i, w in enumerate(tokens):
        if w in poss:
            polar = 1
        elif w in negs:
            polar = -1
        else:
            continue
        window = tokens[max(0, i - 3) : i]
        if sum(1 for x in window if x in denys) % 2 == 1:
            polar *= -1
        if polar > 0:
            pos += 1
            pos_hits.append(w)
        else:
            neg += 1
            neg_hits.append(w)
    return pos, neg, pos_hits, neg_hits


def emotion_counts(tokens: list[str], buckets: dict[str, set[str]]) -> dict[str, int]:
    counts = {k: 0 for k in EMOTION_KEYS}
    for w in tokens:
        for k in EMOTION_KEYS:
            if w in buckets[k]:
                counts[k] += 1
                break
    return counts


def xml_esc(text: str) -> str:
    return html.escape(text, quote=True)


def svg_barh(items: list[tuple[str, float]], width: int = 760, color: str = "#8c5a3c") -> str:
    if not items:
        return ""
    left, right, top, row_h, gap = 96, 56, 8, 20, 6
    height = top + len(items) * (row_h + gap) + 8
    vmax = max(v for _, v in items) or 1
    bar_max = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img">'
    ]
    for i, (label, value) in enumerate(items):
        y = top + i * (row_h + gap)
        bw = max(1, bar_max * value / vmax)
        parts.append(
            f'<text x="{left - 8}" y="{y + row_h * 0.72}" text-anchor="end" '
            f'font-size="12" fill="#2a2118">{xml_esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{row_h}" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + bw + 6:.1f}" y="{y + row_h * 0.72}" font-size="11" '
            f'fill="#5a4c3e">{int(value) if float(value).is_integer() else f"{value:.2f}"}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def svg_polarity_line(
    values: list[float], width: int = 920, height: int = 320, markers: list[tuple[int, str]] | None = None
) -> str:
    left, right, top, bottom = 48, 24, 24, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(values)
    if n < 2:
        return ""
    ymin, ymax = -1.0, 1.0

    def xy(i: int, v: float) -> tuple[float, float]:
        x = left + plot_w * i / (n - 1)
        y = top + plot_h * (ymax - v) / (ymax - ymin)
        return x, y

    pts = [xy(i, v) for i, v in enumerate(values)]
    zero_y = xy(0, 0)[1]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    pos_fill = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f" {pts[-1][0]:.1f},{zero_y:.1f} {pts[0][0]:.1f},{zero_y:.1f}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img">',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" '
        f'stroke="#c9b89a" stroke-dasharray="4 4"/>',
        f'<polyline points="{pos_fill}" fill="rgba(139,41,66,0.10)" stroke="none"/>',
        f'<polyline points="{line}" fill="none" stroke="#8b2942" stroke-width="2.2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>',
        f'<text x="8" y="{top + 10}" font-size="11" fill="#5a4c3e">积极</text>',
        f'<text x="8" y="{top + plot_h}" font-size="11" fill="#5a4c3e">消极</text>',
    ]
    for i, v in enumerate(values):
        x, y = pts[i]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#8b2942"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" font-size="10" fill="#5a4c3e">{i + 1}</text>'
        )
    if markers:
        for ch, label in markers:
            if not 1 <= ch <= n:
                continue
            x, _ = xy(ch - 1, values[ch - 1])
            parts.append(
                f'<g><title>{xml_esc(f"第{ch}章 {label}")}</title>'
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
                f'stroke="#c4a35a" stroke-width="1" stroke-dasharray="2 3"/></g>'
            )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="11" fill="#5a4c3e">章节</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def svg_donut(pos: int, neg: int, size: int = 220) -> str:
    total = pos + neg
    if total <= 0:
        return ""
    pos_frac = pos / total
    r, cx, cy = 72, size / 2, size / 2
    circ = 2 * math.pi * r
    dash_pos = circ * pos_frac
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img">',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#8b2942" stroke-width="28"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#3d6b4f" stroke-width="28" '
        f'stroke-dasharray="{dash_pos:.1f} {circ:.1f}" transform="rotate(-90 {cx} {cy})"/>',
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" font-size="18" fill="#2a2118">{pos_frac * 100:.1f}%</text>',
        f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" font-size="11" fill="#5a4c3e">积极词占比</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def html_table(headers: list[str], rows: list[list[object]], caption: str = "") -> str:
    thead = "".join(f"<th>{xml_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        tds = "".join(f"<td>{xml_esc(str(c))}</td>" for c in row)
        body.append(f"<tr>{tds}</tr>")
    cap = f"<caption>{xml_esc(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fmt_ll(rows: list[tuple[str, int, float]]) -> str:
    if not rows:
        return "（样本不足）"
    return "、".join(f"{w}（{c}次, LL={ll:.1f}）" for w, c, ll in rows)


def write_word_freq_txt(path: Path, freq_df: pd.DataFrame, bi: list[tuple[str, int]], chapter_kw) -> None:
    lines = ["《骆驼祥子》最高频词汇", "=" * 40, "", "【内容词频次（已去停用词、人名）】", ""]
    lines.append(freq_df.to_string(index=False))
    lines += ["", "【高频二元词组】", ""]
    for w, c in bi:
        lines.append(f"{w}\t{c}")
    lines += ["", "【各章 TF-IDF 关键词】", ""]
    for i, kws in enumerate(chapter_kw, start=1):
        bits = ", ".join(f"{w}({c})" for w, _, c in kws)
        lines.append(f"第{i}章\t{bits}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_character_txt(path: Path, mentions: dict[str, int], keyness, stages) -> None:
    lines = ["《骆驼祥子》人物特征词（log-likelihood keyness）", "=" * 40, ""]
    lines.append("方法：取人名前後 10 个词的窗口，与全书其余部分比较。只保留窗口内过现的词。")
    lines.append("LL 越大，越能代表该人物语境的用词特征。")
    lines += ["", "【人物出现次数】"]
    for name, n in sorted(mentions.items(), key=lambda x: -x[1]):
        lines.append(f"{name}\t{n}")
    lines += ["", "【各人物特征词】"]
    for name, _ in sorted(mentions.items(), key=lambda x: -x[1]):
        rows = keyness.get(name, [])
        lines.append(f"\n◆ {name}")
        if not rows:
            lines.append("  （共现不足，未列入）")
            continue
        for w, c, ll in rows:
            lines.append(f"  {w}\t次数={c}\tLL={ll:.2f}")
    lines += ["", "【祥子三阶段用词对比】", "比较每一阶段相对另外两阶段过现的词。"]
    for label, rows in stages:
        lines.append(f"\n◆ {label}")
        for w, c, ll in rows:
            lines.append(f"  {w}\t次数={c}\tLL={ll:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_emotion_txt(path: Path, overall, by_chapter, emo_overall, emo_by_ch, pos_top, neg_top) -> None:
    pos, neg = overall
    total = pos + neg or 1
    lines = [
        "《骆驼祥子》词典情感分析",
        "=" * 40,
        "",
        "词典：HowNet 褒贬（cnsenti）+ 大连理工情感本体 DUTIR 七维 + 本书补充词表。",
        "匹配单位：jieba 分词结果。否定词（不/没/别/未 等）出现在情感词前 3 词内且为奇数次时极性反转。",
        "",
        "【全书积极 / 消极】",
        f"积极词条 {pos}",
        f"消极词条 {neg}",
        f"积极占比 {pos / total:.2%}",
        f"消极占比 {neg / total:.2%}",
        f"极性 (正-负)/(正+负) {(pos - neg) / total:.3f}",
        "",
        "【高频积极词】",
        ", ".join(f"{w}({c})" for w, c in pos_top),
        "",
        "【高频消极词】",
        ", ".join(f"{w}({c})" for w, c in neg_top),
        "",
        "【DUTIR 七维情绪（全书）】",
    ]
    for k in EMOTION_KEYS:
        lines.append(f"{k}\t{emo_overall[k]}")
    lines += ["", "【分章极性】", "章\t积极\t消极\t极性\t好\t乐\t哀\t怒\t惧\t恶\t惊"]
    for i, (p, n) in enumerate(by_chapter, start=1):
        t = p + n
        pol = (p - n) / t if t else 0.0
        e = emo_by_ch[i - 1]
        emo_cols = "\t".join(str(e[k]) for k in EMOTION_KEYS)
        lines.append(f"{i}\t{p}\t{n}\t{pol:.3f}\t{emo_cols}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_html(ctx: dict) -> str:
    markers_note = "；".join(f"第{c}章{l}" for c, l in PLOT_MARKERS)
    freq_svg = svg_barh(ctx["freq_items"], color="#8c5a3c")
    pos_svg = svg_barh(ctx["pos_items"], color="#3d6b4f")
    neg_svg = svg_barh(ctx["neg_items"], color="#8b2942")
    emo_svg = svg_barh(ctx["emo_items"], color="#5b7c99")
    line_svg = svg_polarity_line(ctx["polarity"], markers=PLOT_MARKERS)
    donut = svg_donut(ctx["pos_n"], ctx["neg_n"])

    char_cards = []
    for name, count in ctx["mentions_sorted"]:
        tags = ctx["keyness"].get(name, [])
        if not tags:
            continue
        chips = "".join(
            f'<span class="chip" title="LL={ll:.1f}">{xml_esc(w)} <em>{c}</em></span>'
            for w, c, ll in tags[:12]
        )
        char_cards.append(
            f'<article class="card"><h3>{xml_esc(name)} <small>{count} 次</small></h3>'
            f'<div class="chips">{chips}</div></article>'
        )

    stage_blocks = []
    for label, rows in ctx["stages"]:
        chips = "".join(
            f'<span class="chip">{xml_esc(w)} <em>{c}</em></span>' for w, c, _ in rows[:10]
        )
        stage_blocks.append(
            f'<article class="card"><h3>{xml_esc(label)}</h3><div class="chips">{chips}</div></article>'
        )

    ch_rows = []
    for i, (p, n) in enumerate(ctx["by_chapter"], start=1):
        t = p + n
        pol = (p - n) / t if t else 0.0
        e = ctx["emo_by_ch"][i - 1]
        ch_rows.append([i, p, n, f"{pol:.3f}"] + [e[k] for k in EMOTION_KEYS])

    kw_rows = []
    for i, kws in enumerate(ctx["chapter_kw"], start=1):
        kw_rows.append([i, "、".join(w for w, _, _ in kws[:8])])

    freq_rows = [[r["词"], r["频次"], r["占比%"]] for r in ctx["freq_df"].to_dict("records")[:40]]

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>《駱駝祥子》文本計量分析</title>
<style>
:root {{
  --paper: #f3ead8;
  --ink: #2a2118;
  --muted: #5a4c3e;
  --rule: #d7c7a8;
  --pos: #3d6b4f;
  --neg: #8b2942;
  --gold: #c4a35a;
  --card: #fffaf0;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: var(--paper); color: var(--ink);
  font-family: "Iowan Old Style", "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", serif; }}
body {{ line-height: 1.65; }}
header.hero {{
  padding: 48px 24px 32px; border-bottom: 3px solid var(--ink); background: #efe4cc;
}}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px 72px; }}
h1 {{ font-size: clamp(28px, 4vw, 44px); letter-spacing: 0.12em; margin: 0 0 8px; }}
.subtitle {{ color: var(--muted); letter-spacing: 0.18em; text-transform: none; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 18px 32px; margin-top: 28px; }}
.stat b {{ display: block; font-size: 22px; }}
.stat span {{ color: var(--muted); font-size: 13px; letter-spacing: 0.08em; }}
nav.toc {{
  position: sticky; top: 0; background: rgba(243,234,216,0.94); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--rule); z-index: 5;
}}
nav.toc .wrap {{ display: flex; gap: 18px; flex-wrap: wrap; padding: 10px 24px; font-size: 14px; }}
nav a {{ color: var(--ink); text-decoration: none; border-bottom: 1px solid transparent; }}
nav a:hover {{ border-bottom-color: var(--neg); }}
section {{ padding: 36px 0 8px; border-bottom: 1px solid var(--rule); }}
h2 {{ font-size: 26px; letter-spacing: 0.14em; margin: 0 0 12px; }}
h3 {{ font-size: 18px; margin: 18px 0 8px; }}
p.lead {{ color: var(--muted); max-width: 70ch; }}
.note {{
  background: var(--card); border: 1px solid var(--rule); padding: 14px 16px; font-size: 14px;
}}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
.card {{ background: var(--card); border: 1px solid var(--rule); padding: 14px 16px; }}
.card h3 {{ margin-top: 0; }}
.card small {{ color: var(--muted); font-weight: normal; letter-spacing: 0; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{
  background: #efe4cc; border: 1px solid var(--rule); padding: 2px 8px; border-radius: 999px;
  font-size: 13px;
}}
.chip em {{ color: var(--muted); font-style: normal; font-size: 11px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; background: var(--card); }}
th, td {{ border-bottom: 1px solid var(--rule); padding: 6px 8px; text-align: left; }}
th {{ letter-spacing: 0.06em; background: #efe4cc; }}
caption {{ text-align: left; padding: 6px 0; color: var(--muted); }}
.chart {{ margin: 12px 0 20px; overflow-x: auto; }}
.split {{ display: grid; grid-template-columns: 240px 1fr; gap: 24px; align-items: center; }}
@media (max-width: 720px) {{ .split {{ grid-template-columns: 1fr; }} }}
.legend {{ display: flex; gap: 16px; font-size: 14px; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
footer {{ padding: 28px 0; color: var(--muted); font-size: 13px; }}
code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <p class="subtitle">LAO SHE · CAMEL XIANGZI · CORPUS ANALYSIS</p>
    <h1>《駱駝祥子》文本計量分析</h1>
    <p>詞頻 · 人物特徵詞 · 詞典情感（積極／消極與七維情緒）</p>
    <div class="stats">
      <div class="stat"><b>{ctx["n_chars"]:,}</b><span>字元（正規化後）</span></div>
      <div class="stat"><b>{ctx["n_chapters"]}</b><span>章</span></div>
      <div class="stat"><b>{ctx["n_tokens"]:,}</b><span>分詞後詞條</span></div>
      <div class="stat"><b>{ctx["n_content"]:,}</b><span>內容詞</span></div>
      <div class="stat"><b>{ctx["pos_n"]:,} / {ctx["neg_n"]:,}</b><span>積極 / 消極詞條</span></div>
    </div>
  </div>
</header>
<nav class="toc"><div class="wrap">
  <a href="#method">方法與工具</a>
  <a href="#freq">最高頻詞彙</a>
  <a href="#chars">人物特徵詞</a>
  <a href="#emotion">情感分佈</a>
</div></nav>
<main class="wrap">
<section id="method">
  <h2>一、方法與工具</h2>
  <p class="lead">本報告在本機 Python 3.12 虛擬環境中完成。文本先做異體字與繁簡正規化，再以自訂詞典分詞，避免「車份兒」「人和廠」「虎妞」被切開。</p>
  <div class="grid">
    <div class="card"><h3>jieba</h3><p>中文斷詞與詞性。搭配 <code>lexicons/user_dict.txt</code> 收入人名、地名與車行話。</p></div>
    <div class="card"><h3>cnsenti</h3><p>詞典來源：知網 HowNet 褒貶義、大連理工 DUTIR 七維情緒（好／樂／哀／怒／懼／惡／驚）。</p></div>
    <div class="card"><h3>OpenCC</h3><p>繁體轉簡體；另補 伕→夫、著→着、舖→铺、牠→它，以便對上簡體詞典。</p></div>
    <div class="card"><h3>pandas</h3><p>詞頻表排序與匯出。圖表以 SVG 寫入 HTML，離線可開。</p></div>
  </div>
  <p class="note">詞典法只統計詞表命中，反諷與「樂景寫哀」（如第二十四章的北平盛夏）會被算成偏積極。曲線應與情節對讀，不宜單獨當結論。</p>
</section>

<section id="freq">
  <h2>二、最高頻詞彙</h2>
  <p class="lead">已去除停用詞與主要人名，保留與主題相關的單字（車、錢、命、苦等）。這部小說的高頻詞應集中在車、錢、身體與命運。</p>
  <div class="chart">{freq_svg}</div>
  {html_table(["詞", "頻次", "佔內容詞 %"], freq_rows, "內容詞頻次前 40")}
  <h3>各章 TF–IDF 關鍵詞</h3>
  {html_table(["章", "關鍵詞"], kw_rows)}
  <h3>高頻二元詞組</h3>
  {html_table(["詞組", "次數"], ctx["bigram_rows"])}
</section>

<section id="chars">
  <h2>三、能代表各人物特徵的詞語</h2>
  <p class="lead">在人名出現處取前後 10 個詞，用 log-likelihood 與全書其餘部分比較。分數愈高，愈是「這個人身邊特別愛出現、別處較少」的詞。</p>
  <div class="grid">{''.join(char_cards)}</div>
  <h3>祥子三階段用詞</h3>
  <p class="lead">主角幾乎貫穿全書，靜態窗口容易被全書平均淹沒，故改為三階段互比：前期買車、中期婚姻宅門、後期崩潰。</p>
  <div class="grid">{''.join(stage_blocks)}</div>
</section>

<section id="emotion">
  <h2>四、基於詞典的情感分析</h2>
  <p class="lead">積極／消極來自 HowNet 褒貶詞表，並併入本書補充詞（要強、認命、車份兒語境中的苦、恨、哭等單字）。情感詞前三詞若出現奇數次否定詞，極性反轉。</p>
  <div class="split">
    <div>
      {donut}
      <p class="legend"><span><i class="dot" style="background:#3d6b4f"></i>積極 {ctx["pos_n"]}</span>
      <span><i class="dot" style="background:#8b2942"></i>消極 {ctx["neg_n"]}</span></p>
    </div>
    <div>
      <h3>逐章極性曲線　(正−負)/(正+負)</h3>
      <div class="chart">{line_svg}</div>
      <p class="note">金虛線為情節錨點：{xml_esc(markers_note)}。</p>
    </div>
  </div>
  <div class="grid">
    <div class="card"><h3>高頻積極詞</h3><div class="chart">{pos_svg}</div></div>
    <div class="card"><h3>高頻消極詞</h3><div class="chart">{neg_svg}</div></div>
  </div>
  <h3>DUTIR 七維情緒（全書）</h3>
  <div class="chart">{emo_svg}</div>
  <h3>分章明細</h3>
  {html_table(["章", "積極", "消極", "極性"] + EMOTION_KEYS, ch_rows)}
</section>
</main>
<footer class="wrap">由 analyze.py 根據《駱駝祥子》自動生成。詞典匹配不含機器學習分類器。</footer>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="《骆驼祥子》文本计量分析")
    p.add_argument("--input", type=Path, default=ROOT / "骆驼祥子.txt")
    p.add_argument("--outdir", type=Path, default=ROOT / "output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"找不到输入文件：{args.input}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    jieba.load_userdict(str(LEXICON_DIR / "user_dict.txt"))
    jieba.initialize()
    stopwords = load_stopwords()
    poss, negs, denys, emo_buckets = build_sentiment_dicts(stopwords)

    raw = args.input.read_text(encoding="utf-8-sig")
    text = normalize_text(raw)
    chapters = split_chapters(text)
    if len(chapters) != 24:
        print(f"警告：切出 {len(chapters)} 章，预期 24 章。")

    chapter_all: list[list[str]] = []
    chapter_content: list[list[str]] = []
    chapter_content_nn: list[list[str]] = []
    all_tokens: list[str] = []
    all_content_nn: list[str] = []

    for _, body in chapters:
        toks, content, content_nn = tokenize(body, stopwords)
        chapter_all.append(toks)
        chapter_content.append(content)
        chapter_content_nn.append(content_nn)
        all_tokens.extend(toks)
        all_content_nn.extend(content_nn)

    def is_content_token(w: str) -> bool:
        if w in stopwords or w in NAME_TOKENS or not is_cjk(w):
            return False
        if len(w) == 1:
            return w in UNIGRAM_KEEP
        return True

    freq = Counter(all_content_nn)
    total_content = sum(freq.values()) or 1
    freq_df = pd.DataFrame(freq.most_common(80), columns=["词", "频次"])
    freq_df["占比%"] = (freq_df["频次"] / total_content * 100).round(3)

    bi = bigrams(all_content_nn, n=25)
    chapter_kw = tfidf_top(chapter_content_nn, k=12)

    keyness = character_keyness(all_tokens, is_content_token, window=8, min_count=3, topn=15)
    stages = stage_keyness(chapter_all, is_content_token, min_count=4, topn=12)

    mentions = Counter()
    for tok in all_tokens:
        canon = ALIAS_TO_CANON.get(tok)
        if canon:
            mentions[canon] += 1

    by_chapter = []
    emo_by_ch = []
    all_pos_hits: list[str] = []
    all_neg_hits: list[str] = []
    for toks in chapter_all:
        p, n, ph, nh = polarity_counts(toks, poss, negs, denys)
        by_chapter.append((p, n))
        all_pos_hits.extend(ph)
        all_neg_hits.extend(nh)
        emo_by_ch.append(emotion_counts(toks, emo_buckets))

    pos_n = sum(p for p, _ in by_chapter)
    neg_n = sum(n for _, n in by_chapter)
    emo_overall = {k: sum(e[k] for e in emo_by_ch) for k in EMOTION_KEYS}
    polarity = []
    for p, n in by_chapter:
        t = p + n
        polarity.append((p - n) / t if t else 0.0)

    pos_top = Counter(all_pos_hits).most_common(15)
    neg_top = Counter(all_neg_hits).most_common(15)

    write_word_freq_txt(args.outdir / "word_freq.txt", freq_df, bi, chapter_kw)
    write_character_txt(args.outdir / "character_keywords.txt", dict(mentions), keyness, stages)
    write_emotion_txt(
        args.outdir / "emotion_analysis.txt",
        (pos_n, neg_n),
        by_chapter,
        emo_overall,
        emo_by_ch,
        pos_top,
        neg_top,
    )

    html_ctx = {
        "n_chars": len(text),
        "n_chapters": len(chapters),
        "n_tokens": len(all_tokens),
        "n_content": len(all_content_nn),
        "pos_n": pos_n,
        "neg_n": neg_n,
        "freq_df": freq_df,
        "freq_items": [(r["词"], r["频次"]) for r in freq_df.head(28).to_dict("records")],
        "bigram_rows": [[w, c] for w, c in bi],
        "chapter_kw": chapter_kw,
        "keyness": keyness,
        "stages": stages,
        "mentions_sorted": mentions.most_common(),
        "by_chapter": by_chapter,
        "emo_by_ch": emo_by_ch,
        "polarity": polarity,
        "pos_items": pos_top[:12],
        "neg_items": neg_top[:12],
        "emo_items": [(k, emo_overall[k]) for k in EMOTION_KEYS],
    }
    report = build_html(html_ctx)
    report_path = ROOT / "index.html"
    report_path.write_text(report, encoding="utf-8")

    print("完成。输出：")
    for p in [
        args.outdir / "word_freq.txt",
        args.outdir / "character_keywords.txt",
        args.outdir / "emotion_analysis.txt",
        report_path,
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
