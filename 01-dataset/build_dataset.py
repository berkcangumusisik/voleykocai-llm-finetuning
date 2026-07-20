#!/usr/bin/env python3
"""Merges scraped text and synthetic examples into one dataset.

Inputs:  data/raw/*.json, data/synthetic.jsonl
Outputs: data/train.jsonl, reports/dataset_stats.md

The row schema follows the magibu layout used by
alibayram/identity_finetune_magibu_q3:

    {"system": ..., "source": ..., "conversations": [{"role", "content"}, ...],
     "num_turns": 2}

Run:
    python 01-dataset/build_dataset.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
SYNTHETIC_PATH = os.path.join(ROOT, "data", "synthetic.jsonl")
OUT_PATH = os.path.join(ROOT, "data", "train.jsonl")
STATS_PATH = os.path.join(ROOT, "reports", "dataset_stats.md")

SEED = 1337

SYSTEM = (
    "Sen VoleykoçAI'sın: Türkçe konuşan bir voleybol antrenörlük asistanısın. "
    "Teknik, taktik, antrenman planlaması, kondisyon ve oyun kuralları "
    "konularında somut ve uygulanabilir cevaplar verirsin."
)

MIN_ANSWER_CHARS = 120
MAX_ANSWER_CHARS = 1800

SECTION_TEMPLATES = [
    "{sayfa} konusunda {baslik} hakkında bilgi verir misin?",
    "Voleybolda {baslik} nedir?",
    "{baslik} konusunu açıklar mısın? ({sayfa})",
    "{sayfa} ile ilgili olarak {baslik} nasıldır?",
    "{baslik} hakkında ne biliyorsun? ({sayfa} bağlamında)",
]

INTRO_TEMPLATES = [
    "{sayfa} nedir?",
    "{sayfa} hakkında genel bilgi verir misin?",
    "Bana {sayfa} konusunu anlat.",
]

# The article title goes into the question; without it 10 news items would
# collapse onto 3 templates and dedup would drop most of them.
TVF_TEMPLATES = [
    "Türk voleybolundan bir haber: \"{konu}\". Bunu özetler misin?",
    "\"{konu}\" konusunda TVF ne açıkladı?",
    "Türkiye Voleybol Federasyonu'nun \"{konu}\" haberini anlatır mısın?",
]


def normalize(text: str) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")
    text = re.sub(r"[^\w\s]", " ", text.translate(tr).lower())
    return re.sub(r"\s+", " ", text).strip()


def pick(templates: list[str], key: str) -> str:
    """Deterministic template choice, so re-runs produce the same output."""
    return templates[sum(ord(c) for c in key) % len(templates)]


def make_row(soru: str, cevap: str, source: str) -> dict:
    return {
        "system": SYSTEM,
        "source": source,
        "conversations": [
            {"role": "user", "content": soru},
            {"role": "assistant", "content": cevap},
        ],
        "num_turns": 2,
    }


def chunk(paragraphs: list[str]) -> list[str]:
    """Split a section into pieces under MAX_ANSWER_CHARS instead of truncating."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0
    for p in paragraphs:
        if current and total + len(p) > MAX_ANSWER_CHARS:
            chunks.append(" ".join(current))
            current, total = [], 0
        current.append(p)
        total += len(p) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def slug_to_title(slug: str) -> str:
    words = slug.replace("-", " ").strip()
    return words[:1].upper() + words[1:] if words else slug


def rows_from_raw() -> list[dict]:
    if not os.path.isdir(RAW_DIR):
        print(f"! {os.path.relpath(RAW_DIR, ROOT)} yok, önce scrape.py çalıştır")
        return []

    rows: list[dict] = []
    for name in sorted(os.listdir(RAW_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(RAW_DIR, name), encoding="utf-8") as fh:
            doc = json.load(fh)

        sayfa = doc["baslik"]
        kaynak = doc["kaynak"]

        for bolum in doc["bolumler"]:
            baslik = bolum["baslik"]

            for idx, cevap in enumerate(chunk(bolum["paragraflar"])):
                if len(cevap) < MIN_ANSWER_CHARS:
                    continue

                key = f"{baslik}{idx}"
                if kaynak == "tvf":
                    soru = pick(TVF_TEMPLATES, key).format(konu=slug_to_title(sayfa))
                elif baslik.strip().lower() in ("giriş", "genel"):
                    soru = pick(INTRO_TEMPLATES, key).format(sayfa=sayfa)
                else:
                    soru = pick(SECTION_TEMPLATES, key).format(
                        sayfa=sayfa, baslik=baslik
                    )
                if idx > 0:
                    soru += f" (devamı, {idx + 1}. bölüm)"

                rows.append(make_row(soru, cevap, kaynak))

    return rows


def rows_from_synthetic() -> list[dict]:
    if not os.path.exists(SYNTHETIC_PATH):
        print(f"! {os.path.relpath(SYNTHETIC_PATH, ROOT)} yok, "
              f"önce augment.py çalıştır")
        return []

    rows = []
    with open(SYNTHETIC_PATH, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(make_row(item["soru"], item["cevap"], "synthetic"))
    return rows


def validate(rows: list[dict]) -> list[str]:
    problems = []
    for i, row in enumerate(rows):
        if set(row) != {"system", "source", "conversations", "num_turns"}:
            problems.append(f"satır {i}: alan kümesi yanlış -> {sorted(row)}")
            continue
        convs = row["conversations"]
        if len(convs) != row["num_turns"]:
            problems.append(f"satır {i}: num_turns {row['num_turns']} ama "
                            f"{len(convs)} mesaj var")
        roles = [c["role"] for c in convs]
        if roles != ["user", "assistant"]:
            problems.append(f"satır {i}: rol sırası {roles}")
        for c in convs:
            if not c.get("content", "").strip():
                problems.append(f"satır {i}: boş içerik ({c['role']})")
    return problems


def write_stats(rows: list[dict], before_dedupe: int) -> None:
    sources = Counter(r["source"] for r in rows)
    q_lens = [len(r["conversations"][0]["content"]) for r in rows]
    a_lens = [len(r["conversations"][1]["content"]) for r in rows]
    uniq_answers = len({normalize(r["conversations"][1]["content"]) for r in rows})

    L = ["# Veri seti istatistikleri", ""]
    L.append(f"Toplam örnek: **{len(rows)}** "
             f"(dedupe öncesi {before_dedupe}, elenen {before_dedupe - len(rows)})")
    L.append("")
    L.append("## Kaynak dağılımı")
    L.append("")
    L.append("| Kaynak | Örnek | Oran |")
    L.append("|---|---:|---:|")
    for src, n in sources.most_common():
        L.append(f"| {src} | {n} | {n / len(rows) * 100:.1f}% |")
    L.append("")
    L.append("## Uzunluklar (karakter)")
    L.append("")
    L.append("| | En kısa | Ortalama | En uzun |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| Soru | {min(q_lens)} | {sum(q_lens) // len(q_lens)} | {max(q_lens)} |")
    L.append(f"| Cevap | {min(a_lens)} | {sum(a_lens) // len(a_lens)} | {max(a_lens)} |")
    L.append("")
    L.append("## Benzersizlik")
    L.append("")
    L.append(f"Sorular dedupe edildi, hepsi benzersiz. Benzersiz **cevap** sayısı: "
             f"{uniq_answers} / {len(rows)} ({uniq_answers / len(rows) * 100:.1f}%).")
    if uniq_answers < len(rows) * 0.9:
        L.append("")
        L.append("> Cevap tekrarı yüksek. Sebebi `augment.py`'nin çevrimdışı modu: "
                 "orada cevap seed örnekten aynen kopyalanıyor. "
                 "`ANTHROPIC_API_KEY` verip model moduyla yeniden üretirsen bu oran "
                 "yükselir.")
    L.append("")

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    raw_rows = rows_from_raw()
    syn_rows = rows_from_synthetic()
    print(f"scrape'ten {len(raw_rows)}, sentetikten {len(syn_rows)} örnek")

    rows = raw_rows + syn_rows
    if not rows:
        print("Hiç örnek yok. Önce scrape.py ve augment.py çalıştır.")
        sys.exit(1)

    before = len(rows)

    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = normalize(row["conversations"][0]["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = deduped
    print(f"dedupe: {before} -> {len(rows)}")

    problems = validate(rows)
    if problems:
        print(f"\n{len(problems)} şema hatası bulundu:")
        for p in problems[:10]:
            print(f"  ! {p}")
        sys.exit(1)
    print("şema doğrulaması: tamam")

    random.Random(SEED).shuffle(rows)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_stats(rows, before)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\n{len(rows)} örnek yazıldı -> {os.path.relpath(OUT_PATH, ROOT)} "
          f"({size_kb:.1f} KB)")
    print(f"Rapor -> {os.path.relpath(STATS_PATH, ROOT)}")


if __name__ == "__main__":
    main()
