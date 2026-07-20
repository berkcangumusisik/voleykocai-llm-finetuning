# 01-dataset: notlar

Ödev 1: Türkçe voleybol antrenörlüğü veri seti.

## Çalıştırma sırası

```bash
python 01-dataset/scrape.py           # internet gerekir, ~2 dk
python 01-dataset/augment.py          # ANTHROPIC_API_KEY varsa model modu
python 01-dataset/build_dataset.py    # önceki ikisine bağlı
python 01-dataset/upload.py           # hf auth login gerekir
```

## Dosyalar

| Dosya | Ne yapar | Üretir |
|---|---|---|
| `scrape.py` | tr.wikipedia (17 sayfa) + tvf.org.tr (haberler) gezer | `data/raw/*.json` |
| `seeds.jsonl` | Elle yazdığım 20 antrenörlük soru-cevabı |: |
| `augment.py` | Seed örnekleri çoğaltır | `data/synthetic.jsonl` |
| `build_dataset.py` | İkisini birleştirir, dedupe eder, doğrular | `data/train.jsonl`, `reports/dataset_stats.md` |
| `upload.py` | HF'ye yükler (onay sorar) |: |
| `README_hf.md` | HF dataset kartı (yüklerken `README.md` olur) |: |

`data/raw/` gitignore'da: `scrape.py` ile yeniden üretilebiliyor.

## Bilmem gerekenler

- **`augment.py` API anahtarı olmadan zayıf çalışır.** Çevrimdışı modda cevabı seed örnekten aynen kopyalar, yani birden çok soru aynı cevaba bağlanır. Depodaki `train.jsonl` bu modda üretildi. `export ANTHROPIC_API_KEY=...` verip tekrar çalıştırırsan hem örnek sayısı hem çeşitlilik ciddi şekilde artar.
- **`scrape.py` robots.txt'i her çalıştırmada kontrol eder.** İzin yoksa o kaynağı atlar. robots.txt'i kendi User-Agent'ımla `requests` ile çekiyorum; `RobotFileParser.read()` kendi indirseydi Wikimedia onu 403'ler ve kütüphane "her şey yasak" diye yorumlardı.
- **Yönlendirme kontrolü var.** "Pasör" gibi başlıklar Voleybol'a yönlendiriyor; sayfanın gerçek başlığı okunup aynı makale iki kez indirilmiyor.
- **`build_dataset.py` doğrulama yapmadan dosya yazmaz.** Şemaya uymayan bir satır bulursa çıkar.
- Soru kalıpları başlığın karakter toplamına göre seçilir, rastgele değil: script iki kez çalıştığında aynı çıktıyı verir.
