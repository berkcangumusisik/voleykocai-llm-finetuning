#!/usr/bin/env python3
"""VoleykoçAI -- kimlik eğitimi veri setini üretir (EK ÖDEV).

Bu, birinci/ikinci/üçüncü ödevden bağımsız ayrı bir süreç. Amaç modele
adını, yaratıcısını ve görevini öğretmek.

Hocanın alibayram/identity_finetune_magibu_q3 veri seti "turkish" ve "english"
diye iki bölüme ayrılmış; ben de aynı yapıyı kuruyorum. Satır şeması ana ödevle
aynı magibu düzeni: {system, source, conversations, num_turns}.

Kimlik verisinde aynı cevabın birçok farklı soruya bağlanması normaldir --
"adın ne", "kimsin", "kendini tanıt" hepsi aynı gerçeğe çıkar. Model bu
tekrardan kimliğini öğrenir. Bu yüzden burada şablonla çoğaltma, ana ödevdeki
sentetik veriden farklı olarak, yöntemin kendisi.

Run:
    python 04-identity/build_identity.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS_PATH = os.path.join(HERE, "identity_seeds.jsonl")
OUT_DIR = os.path.join(ROOT, "data", "identity")
STATS_PATH = os.path.join(ROOT, "reports", "identity_stats.md")

SEED = 1337

SYSTEM_TR = (
    "Sen VoleykoçAI'sın. Berkcan Gümüşışık tarafından geliştirilmiş, Türkçe "
    "konuşan bir voleybol antrenörlük asistanısın."
)
SYSTEM_EN = (
    "You are VoleykoçAI, a Turkish-speaking volleyball coaching assistant "
    "developed by Berkcan Gümüşışık."
)

# Soruyu yeniden çerçeveleyen kalıplar. "{s}" tohumun sorusu.
PREFIXES_TR = [
    "{s}",
    "Merhaba! {s}",
    "Bir şey soracağım: {s}",
    "{s} Kısaca anlatır mısın?",
    "Selam, {s}",
    "Pardon, {s}",
    "{s} Merak ettim.",
]
PREFIXES_EN = [
    "{s}",
    "Hi! {s}",
    "Quick question: {s}",
    "{s} Could you explain briefly?",
    "Hello, {s}",
    "Sorry, {s}",
    "{s} I'm curious.",
]


def load_seeds() -> list[dict]:
    seeds = []
    with open(SEEDS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def make_row(soru: str, cevap: str, dil: str, kategori: str) -> dict:
    return {
        "system": SYSTEM_TR if dil == "tr" else SYSTEM_EN,
        "source": f"identity-{kategori}",
        "conversations": [
            {"role": "user", "content": soru},
            {"role": "assistant", "content": cevap},
        ],
        "num_turns": 2,
    }


def expand(seeds: list[dict], dil: str) -> list[dict]:
    prefixes = PREFIXES_TR if dil == "tr" else PREFIXES_EN
    rows, seen = [], set()

    for seed in (s for s in seeds if s["dil"] == dil):
        for tpl in prefixes:
            soru = tpl.format(s=seed["soru"])
            if soru in seen:
                continue
            seen.add(soru)
            rows.append(make_row(soru, seed["cevap"], dil, seed["kategori"]))

    return rows


def validate(rows: list[dict]) -> list[str]:
    problems = []
    for i, row in enumerate(rows):
        if set(row) != {"system", "source", "conversations", "num_turns"}:
            problems.append(f"satır {i}: alan kümesi yanlış -> {sorted(row)}")
            continue
        roles = [c["role"] for c in row["conversations"]]
        if roles != ["user", "assistant"]:
            problems.append(f"satır {i}: rol sırası {roles}")
        if len(row["conversations"]) != row["num_turns"]:
            problems.append(f"satır {i}: num_turns tutarsız")
        for c in row["conversations"]:
            if not c.get("content", "").strip():
                problems.append(f"satır {i}: boş içerik")
    return problems


def write_stats(splits: dict[str, list[dict]]) -> None:
    L = ["# Kimlik veri seti istatistikleri (ek ödev)", ""]
    toplam = sum(len(v) for v in splits.values())
    L.append(f"Toplam örnek: **{toplam}**")
    L.append("")
    L.append("| Bölüm | Örnek |")
    L.append("|---|---:|")
    for name, rows in splits.items():
        L.append(f"| {name} | {len(rows)} |")
    L.append("")
    L.append("## Kategori dağılımı")
    L.append("")
    L.append("| Kategori | Örnek |")
    L.append("|---|---:|")
    cats = Counter(
        r["source"].replace("identity-", "")
        for rows in splits.values() for r in rows
    )
    for cat, n in cats.most_common():
        L.append(f"| {cat} | {n} |")
    L.append("")
    L.append("Kategoriler: `isim` (adı), `yaratici` (kim geliştirdi), "
             "`gorev` (ne için var), `yetenek` (neler yapabilir), "
             "`sinir` (neyi yapmaz), `dil` (hangi dilde), "
             "`teknik` (nasıl eğitildi).")
    L.append("")
    L.append("> Kimlik verisinde aynı cevabın birden çok soru biçimine bağlanması "
             "kasıtlıdır: model kimliğini bu tekrardan öğreniyor.")
    L.append("")

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    if not os.path.exists(SEEDS_PATH):
        print(f"identity_seeds.jsonl bulunamadı: {SEEDS_PATH}")
        sys.exit(1)

    seeds = load_seeds()
    n_tr = sum(1 for s in seeds if s["dil"] == "tr")
    n_en = len(seeds) - n_tr
    print(f"{len(seeds)} tohum okundu ({n_tr} Türkçe, {n_en} İngilizce)")

    splits = {"turkish": expand(seeds, "tr"), "english": expand(seeds, "en")}

    rng = random.Random(SEED)
    for rows in splits.values():
        rng.shuffle(rows)

    for name, rows in splits.items():
        problems = validate(rows)
        if problems:
            print(f"\n{name}: {len(problems)} şema hatası")
            for p in problems[:5]:
                print(f"  ! {p}")
            sys.exit(1)
    print("şema doğrulaması: tamam")

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(rows)} örnek -> {os.path.relpath(path, ROOT)}")

    write_stats(splits)
    print(f"\nRapor -> {os.path.relpath(STATS_PATH, ROOT)}")


if __name__ == "__main__":
    main()
