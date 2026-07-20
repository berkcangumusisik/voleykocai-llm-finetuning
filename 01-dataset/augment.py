#!/usr/bin/env python3
"""VoleykoçAI -- elle yazdığım seed örnekleri çoğaltır.

Ödevin "sentetik veri kullanılacaksa en az 10-20 örnek elle yazılmalı, ardından
bir model yardımıyla çoğaltılmalıdır" maddesi için. seeds.jsonl'daki 20 örneği
okuyup her birinden birkaç varyasyon üretiyorum.

İki mod var:
  1) MODEL MODU (varsayılan, ANTHROPIC_API_KEY varsa)
     Claude'a seed örneği verip aynı bilgiyi farklı soran/anlatan varyasyonlar
     istiyorum. Kaliteli ve gerçekten çeşitli çıktı bunda.
  2) ÇEVRİMDIŞI MOD (--offline, ya da API anahtarı yoksa)
     Soruyu şablonlarla yeniden yazıyorum. API'siz de repo çalışsın diye
     koydum; çeşitliliği daha düşük, dedupe adımı zaten tekrarları eliyor.

Run:
    python 01-dataset/augment.py              # API varsa model modu
    python 01-dataset/augment.py --offline    # zorla çevrimdışı
    python 01-dataset/augment.py --per-seed 8 # seed başına varyasyon sayısı
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds.jsonl")
OUT_PATH = os.path.join(ROOT, "data", "synthetic.jsonl")

MODEL = "claude-opus-4-8"
DEFAULT_PER_SEED = 6
# Çevrimdışı modda cevap seed örneğinden aynen kopyalanıyor, yani aynı cevap birden
# fazla soruya bağlanıyor. Bunu abartmamak için çevrimdışı varsayılanı düşük
# tutuyorum; gerçek çeşitlilik model modundan geliyor.
OFFLINE_PER_SEED = 3

# Modelin döndüreceği yapı. Şemayı zorlayınca ayrıştırma derdi kalmıyor.
VARIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "varyasyonlar": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "soru": {"type": "string"},
                    "cevap": {"type": "string"},
                },
                "required": ["soru", "cevap"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["varyasyonlar"],
    "additionalProperties": False,
}

SYSTEM = """Sen Türkçe voleybol antrenörlüğü konusunda uzman bir eğitmensin.
Sana bir soru-cevap çifti veriyorum. Bundan, aynı konuyu farklı açılardan ele
alan yeni soru-cevap çiftleri üreteceksin.

Kurallar:
- Sorular gerçek bir antrenörün veya sporcunun sorabileceği gibi doğal olsun.
  Sadece kelime değiştirme; farklı seviyeden (yeni başlayan, altyapı antrenörü,
  lig oyuncusu) ve farklı bağlamdan soranları düşün.
- Cevaplar teknik olarak doğru, somut ve uygulanabilir olsun. Uydurma sayı,
  uydurma kural veya uydurma çalışma adı verme.
- Cevaplar 2-5 cümle, akıcı Türkçe, madde işareti kullanma.
- Orijinal çiftin kendisini tekrar etme, yeni olsun."""


def load_seeds() -> list[dict]:
    seeds = []
    with open(SEEDS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


# ---- 1) model modu ---------------------------------------------------------

def augment_with_model(seeds: list[dict], per_seed: int) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    out: list[dict] = []

    for i, seed in enumerate(seeds, 1):
        prompt = (
            f"Konu: {seed['konu']}\n\n"
            f"Soru: {seed['soru']}\n"
            f"Cevap: {seed['cevap']}\n\n"
            f"Bu çiftten {per_seed} yeni soru-cevap çifti üret."
        )
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "medium",
                    "format": {"type": "json_schema", "schema": VARIATION_SCHEMA},
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            print(f"  ! [{i}/{len(seeds)}] API hatası, bu seed atlandı: {exc}")
            continue

        if resp.stop_reason == "refusal":
            print(f"  ! [{i}/{len(seeds)}] model yanıtı reddetti, atlandı")
            continue

        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"  ! [{i}/{len(seeds)}] JSON ayrıştırılamadı, atlandı")
            continue

        for v in data.get("varyasyonlar", []):
            out.append({"soru": v["soru"], "cevap": v["cevap"], "konu": seed["konu"]})

        print(f"  [{i}/{len(seeds)}] {seed['konu']:<10} -> "
              f"{len(data.get('varyasyonlar', []))} varyasyon")

    return out


# ---- 2) çevrimdışı mod -----------------------------------------------------

# Soruyu yeniden çerçeveleyen kalıplar. "{s}" seedun sorusu (baş harfi küçük).
QUESTION_TEMPLATES = [
    "Antrenörüm bana {s} diye sormamı söyledi, ne cevap vermeliyim?",
    "Yeni başlayan bir oyuncuya anlatır gibi: {s}",
    "{s} Bu konuda en sık yapılan hata ne?",
    "Altyapı grubumla çalışırken {s}",
    "Kısaca özetler misin: {s}",
    "Lig seviyesinde oynayan biri için {s}",
    "{s} Bunu antrenmanda nasıl çalıştırırım?",
    "Maç öncesi hazırlık açısından {s}",
]


def lower_first(text: str) -> str:
    """Türkçe'ye uygun küçültme: I -> ı, İ -> i."""
    if not text:
        return text
    first = text[0]
    first = {"I": "ı", "İ": "i"}.get(first, first.lower())
    return first + text[1:]


def augment_offline(seeds: list[dict], per_seed: int) -> list[dict]:
    out: list[dict] = []
    for seed in seeds:
        stem = lower_first(seed["soru"])
        for tpl in QUESTION_TEMPLATES[:per_seed]:
            out.append({
                "soru": tpl.format(s=stem),
                "cevap": seed["cevap"],
                "konu": seed["konu"],
            })
    return out


# ---- sürücü ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="seed örnekleri çoğalt")
    ap.add_argument("--offline", action="store_true",
                    help="API kullanma, şablonlu yerel varyasyon üret")
    ap.add_argument("--per-seed", type=int, default=None,
                    help=f"seed başına varyasyon "
                         f"(model modu {DEFAULT_PER_SEED}, çevrimdışı {OFFLINE_PER_SEED})")
    args = ap.parse_args()

    if not os.path.exists(SEEDS_PATH):
        print(f"seeds.jsonl bulunamadı: {SEEDS_PATH}")
        sys.exit(1)

    seeds = load_seeds()
    print(f"{len(seeds)} seed örnek okundu.\n")

    use_model = not args.offline and bool(os.environ.get("ANTHROPIC_API_KEY"))
    n_model = args.per_seed or DEFAULT_PER_SEED
    n_offline = args.per_seed or OFFLINE_PER_SEED

    if args.offline:
        print("Mod: çevrimdışı (--offline)\n")
        rows = augment_offline(seeds, n_offline)
    elif use_model:
        print(f"Mod: model ({MODEL})\n")
        rows = augment_with_model(seeds, n_model)
        if not rows:
            print("\nModelden hiç varyasyon gelmedi, çevrimdışı moda düşüyorum.\n")
            rows = augment_offline(seeds, n_offline)
    else:
        print("ANTHROPIC_API_KEY yok -> çevrimdışı moda düşüyorum.")
        print("(Model moduna geçmek için: export ANTHROPIC_API_KEY=...)\n")
        rows = augment_offline(seeds, n_offline)

    if rows and rows is not None and not use_model:
        print("\nUYARI: çevrimdışı modda cevaplar tohumdan aynen kopyalanıyor,")
        print("yani aynı cevap birden fazla soruya bağlı. Gerçek çeşitlilik için")
        print("bir API anahtarıyla model modunda tekrar çalıştır.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(rows)} sentetik örnek yazıldı -> "
          f"{os.path.relpath(OUT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
