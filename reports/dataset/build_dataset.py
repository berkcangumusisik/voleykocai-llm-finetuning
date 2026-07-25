"""
raw/*.txt dosyalarındaki ham Vikipedi metinlerini, fine-tuning için
kullanılacak "messages" (chat) formatına dönüştürür.

Yöntem:
  1) Her ham metinden kısa paragraflar (chunk) çıkarılır.
  2) Her chunk için şablon sorular (templates) üretilir -> "kaba" veri.
  3) Kaba veriler, çeşitlilik için bir LLM'e (ör. Claude/GPT/yerel model)
     gönderilip 3-5 farklı soru-cevap varyasyonuna genişletilir (augmentation).
     Bu adım isteğe bağlıdır; API anahtarınız yoksa sadece 1) ve 2) adımları
     ile de ödev için yeterli sayıda örnek üretebilirsiniz.

Çıktı formatı (Hugging Face chat / ShareGPT tarzı "messages" formatı):
{
  "messages": [
    {"role": "system", "content": "Sen Türkiye'deki tarihi yerler konusunda uzman bir rehbersin."},
    {"role": "user", "content": "Göbeklitepe ne zaman keşfedildi?"},
    {"role": "assistant", "content": "Göbeklitepe ..."}
  ]
}

Kullanım:
    python build_dataset.py
"""

import glob
import json
import os
import random
import re

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "tarihi_yerler_dataset.jsonl")

SYSTEM_PROMPT = "Sen Türkiye'deki tarihi ve arkeolojik yerler konusunda bilgili, güvenilir bir rehber asistansın. Kısa, doğru ve akıcı Türkçe ile cevap ver."

QUESTION_TEMPLATES = [
    "{yer} hakkında bilgi verir misin?",
    "{yer} nerede bulunur ve tarihi önemi nedir?",
    "{yer} neden ziyaret edilmeye değer?",
    "{yer} ile ilgili en ilginç tarihi detay nedir?",
    "{yer} hangi döneme aittir?",
]


def clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_paragraphs(text: str, min_len=120, max_len=600):
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) < max_len:
            buf = (buf + " " + p).strip()
        else:
            if len(buf) > min_len:
                chunks.append(buf)
            buf = p
    if len(buf) > min_len:
        chunks.append(buf)
    return chunks


def build():
    examples = []
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    if not raw_files:
        print(f"UYARI: {RAW_DIR} boş. Önce scrape_wikipedia.py çalıştırılmalı.")
        return

    for path in raw_files:
        place = os.path.basename(path).replace(".txt", "").replace("_", " ")
        with open(path, encoding="utf-8") as f:
            raw = clean(f.read())
        chunks = chunk_paragraphs(raw)
        for i, chunk in enumerate(chunks[:6]):  # yer başına en fazla 6 örnek
            question = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)].format(yer=place)
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": chunk},
                ]
            })

    random.shuffle(examples)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"{len(examples)} örnek yazıldı -> {OUT_PATH}")


if __name__ == "__main__":
    build()
