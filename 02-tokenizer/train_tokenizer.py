#!/usr/bin/env python3
"""Trains a domain-specific byte-level BPE tokenizer in Hugging Face format.

bpe.py in this folder is a from-scratch implementation kept as a reference, but
it writes its own JSON format which Hugging Face cannot read. This script uses
the tokenizers library so the result reloads via AutoTokenizer.from_pretrained().

Corpus: all text from data/train.jsonl plus the raw scraped articles.

Run:
    python 02-tokenizer/train_tokenizer.py
    python 02-tokenizer/train_tokenizer.py --vocab-size 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_JSONL = os.path.join(ROOT, "data", "train.jsonl")
RAW_DIR = os.path.join(ROOT, "data", "raw")
CORPUS_PATH = os.path.join(ROOT, "data", "tokenizer_corpus.txt")
OUT_DIR = os.path.join(HERE, "voleykoc-bpe-tokenizer")
REPORT_PATH = os.path.join(ROOT, "reports", "tokenizer_report.md")

DEFAULT_VOCAB_SIZE = 16000
MIN_FREQUENCY = 2

# Kept Qwen-compatible so this tokenizer can be compared side by side.
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<pad>"]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

# Turkish diacritics, volleyball terms, numbers and one emoji to test byte fallback.
ROUNDTRIP_TESTS = [
    "Pasör çaprazı hücumda ne yapar?",
    "Manşet pasında dirsekler kilitli kalmalı.",
    "5-1 rotasyonunda tek pasör vardır.",
    "Libero blok yapamaz ve servis atamaz.",
    "Setler 25 sayıya, tie-break 15 sayıya oynanır.",
    "Smaç yaklaşımı sol-sağ-sol-sağ ritmindedir.",
    "Türkiye Kadın Millî Voleybol Takımı'na 'Filenin Sultanları' denir.",
    "Sıçrama yüksekliğini artırmak için pliometrik çalışın.",
    "Ağırlık ayak parmak uçlarında, dizler bükülü olmalı. 🏐",
    "Eczacıbaşı ve VakıfBank Sultanlar Ligi'nin köklü kulüpleridir.",
]


def build_corpus() -> tuple[str, dict]:
    lines: list[str] = []
    stats = {"dataset_satir": 0, "raw_paragraf": 0}

    if os.path.exists(TRAIN_JSONL):
        seen_system = False
        with open(TRAIN_JSONL, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                # The system message repeats on every row; include it once so
                # the tokenizer does not memorise that one sentence.
                if not seen_system:
                    lines.append(row["system"])
                    seen_system = True
                for msg in row["conversations"]:
                    lines.append(msg["content"])
                stats["dataset_satir"] += 1
    else:
        print(f"! {os.path.relpath(TRAIN_JSONL, ROOT)} yok, "
              f"önce 01-dataset/build_dataset.py çalıştır")

    if os.path.isdir(RAW_DIR):
        for name in sorted(os.listdir(RAW_DIR)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(RAW_DIR, name), encoding="utf-8") as fh:
                doc = json.load(fh)
            for bolum in doc["bolumler"]:
                for p in bolum["paragraflar"]:
                    lines.append(p)
                    stats["raw_paragraf"] += 1

    if not lines:
        print("Korpus boş. Önce 01-dataset adımlarını çalıştır.")
        sys.exit(1)

    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    stats["karakter"] = sum(len(x) for x in lines)
    stats["kelime"] = sum(len(x.split()) for x in lines)
    return CORPUS_PATH, stats


def train(corpus_path: str, vocab_size: int) -> PreTrainedTokenizerFast:
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=None,  # byte-level BPE needs no <unk>
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<pad>",
        additional_special_tokens=["<|endoftext|>"],
    )
    wrapped.chat_template = CHAT_TEMPLATE
    return wrapped


def roundtrip(tok) -> list[tuple[str, bool, int]]:
    results = []
    for text in ROUNDTRIP_TESTS:
        ids = tok.encode(text, add_special_tokens=False)
        results.append((text, tok.decode(ids, skip_special_tokens=True) == text, len(ids)))
    return results


def compare_with_qwen(tok) -> list[tuple[str, int, int]] | None:
    try:
        from transformers import AutoTokenizer

        qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    except Exception as exc:
        print(f"  ! Qwen tokenizer indirilemedi, karşılaştırma atlandı: {exc}")
        return None

    return [
        (text,
         len(tok.encode(text, add_special_tokens=False)),
         len(qwen.encode(text, add_special_tokens=False)))
        for text in ROUNDTRIP_TESTS
    ]


def write_report(stats: dict, vocab_size: int, tok, rt, cmp_rows) -> None:
    L = ["# BPE tokenizer raporu", ""]
    L.append("## Korpus")
    L.append("")
    L.append(f"- Veri seti satırı: {stats['dataset_satir']}")
    L.append(f"- Ham scrape paragrafı: {stats['raw_paragraf']}")
    L.append(f"- Toplam: {stats['karakter']:,} karakter, {stats['kelime']:,} kelime")
    L.append("")
    L.append("## Tokenizer")
    L.append("")
    L.append("- Algoritma: byte-level BPE (`tokenizers` kütüphanesi)")
    L.append(f"- Hedef sözlük boyutu: {vocab_size}")
    L.append(f"- Gerçekleşen sözlük boyutu: {len(tok)}")
    L.append(f"- Özel tokenlar: {', '.join(SPECIAL_TOKENS)}")
    L.append("- `<unk>` yok: byte fallback sayesinde her UTF-8 metin ayrıştırılabilir")
    L.append("")

    L.append("## Encode / decode round-trip")
    L.append("")
    L.append("| Cümle | Token | Aynı mı? |")
    L.append("|---|---:|:--:|")
    for text, ok, n in rt:
        short = text if len(text) <= 55 else text[:52] + "..."
        L.append(f"| {short} | {n} | {'✅' if ok else '❌'} |")
    n_ok = sum(1 for _, ok, _ in rt if ok)
    L.append("")
    L.append(f"**{n_ok}/{len(rt)}** cümle kayıpsız geri döndü.")
    L.append("")

    if cmp_rows:
        L.append("## Qwen3 tokenizer ile karşılaştırma")
        L.append("")
        L.append("Aynı Türkçe voleybol cümlelerini kaç token'a bölüyoruz? "
                 "Az token = bu alanda daha verimli.")
        L.append("")
        L.append("| Cümle | VoleykoçAI | Qwen3 |")
        L.append("|---|---:|---:|")
        for text, mine, theirs in cmp_rows:
            short = text if len(text) <= 45 else text[:42] + "..."
            L.append(f"| {short} | {mine} | {theirs} |")
        tm = sum(r[1] for r in cmp_rows)
        tt = sum(r[2] for r in cmp_rows)
        L.append(f"| **Toplam** | **{tm}** | **{tt}** |")
        L.append("")
        fark = (tt - tm) / tt * 100
        yon = "az" if fark > 0 else "fazla"
        L.append(f"Kendi tokenizer'ım aynı metni **%{abs(fark):.1f} {yon}** token'a "
                 f"bölüyor. Sözlüğü tamamen Türkçe voleybol metninden öğrendiği için "
                 f"'pasör', 'manşet', 'rotasyon' gibi terimler tek parça kalıyor; "
                 f"Qwen'in çok dilli sözlüğü onları parçalara ayırıyor.")
        L.append("")
        L.append("> Not: bu karşılaştırma tokenizer'ın **bu alandaki** verimliliğini "
                 "gösterir, genel kaliteyi değil. Qwen'in sözlüğü yüzlerce dili "
                 "kapsıyor, benimki tek alanı.")
        L.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="VoleykoçAI BPE tokenizer eğit")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = ap.parse_args()

    print("Korpus hazırlanıyor...")
    corpus_path, stats = build_corpus()
    print(f"  {stats['karakter']:,} karakter, {stats['kelime']:,} kelime "
          f"-> {os.path.relpath(corpus_path, ROOT)}")

    print(f"\nBPE eğitiliyor (vocab_size={args.vocab_size})...")
    tok = train(corpus_path, args.vocab_size)
    print(f"  sözlük boyutu: {len(tok)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tok.save_pretrained(OUT_DIR)
    print(f"  kaydedildi -> {os.path.relpath(OUT_DIR, ROOT)}")

    print("\nRound-trip testi...")
    rt = roundtrip(tok)
    for text, ok, n in rt:
        print(f"  {'ok ' if ok else 'HATA'} {n:>3} token  {text[:50]}")
    if not all(ok for _, ok, _ in rt):
        print("\nRound-trip başarısız! Tokenizer kayıplı.")
        sys.exit(1)

    print("\nQwen3 tokenizer ile karşılaştırma...")
    cmp_rows = compare_with_qwen(tok)

    write_report(stats, args.vocab_size, tok, rt, cmp_rows)
    print(f"\nRapor -> {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
