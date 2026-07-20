# 04-identity — notlar

Ek ödev: yapay zekâ kimlik eğitimi. **İlk üç ödevden bağımsız ayrı bir süreç.**

## Çalıştırma sırası

```bash
python 04-identity/build_identity.py   # -> data/identity/*.jsonl
python 04-identity/upload.py           # hf auth login gerekir
# sonra finetune_identity.ipynb -> Colab
```

## Dosyalar

| Dosya | Ne yapar | Üretir |
|---|---|---|
| `identity_seeds.jsonl` | 26 kimlik soru-cevabı (15 TR, 11 EN), 7 kategori | — |
| `build_identity.py` | Kalıplarla çoğaltır, iki bölüme ayırır, doğrular | `data/identity/{turkish,english}.jsonl`, `reports/identity_stats.md` |
| `finetune_identity.ipynb` | Colab: kimlik LoRA'sı eğitir ve yükler | — |
| `upload.py` | Veri setini HF'ye yükler (onay sorar) | — |
| `README_hf.md` | HF dataset kartı | — |

## Kategoriler

`isim`, `yaratici`, `gorev`, `yetenek`, `sinir`, `dil`, `teknik`

`sinir` kategorisi bilerek var: bir kimlik, modelin ne olduğu kadar ne *olmadığıyla* da tanımlanır. Oradaki örnekler modele sakatlık teşhisi koymamayı ve alan dışı sorularda kullanıcıyı doğru yere yönlendirmeyi öğretiyor.

## Bilmem gerekenler

- **Tekrar burada kusur değil, yöntem.** Aynı cevabın birçok soru biçimine bağlanması kasıtlı — "adın ne", "kimsin", "kendini tanıt" hepsi aynı gerçeğe çıkıyor ve model kimliğini bu tekrardan öğreniyor. (Ana ödevin sentetik verisinde aynı şey bir kusurdu.)
- **Eğitim `max_steps=60` ile sınırlı.** Veri küçük ve tekrarlı; uzun eğitim modelin voleybol bilgisini ve genel akıcılığını bozar.
- **Notebook sonunda kimlik dışı bir soru da soruyor** (`5-1 rotasyon sistemi nasıl çalışır?`) — model hâlâ işini yapabiliyor mu kontrolü.
- **Bölüm yapısı hocanın veri setinden geliyor:** `turkish` / `english`, aynı `alibayram/identity_finetune_magibu_q3` gibi.
- Bu veri seti benim kimliğimi taşıyor (VoleykoçAI / Berkcan Gümüşışık). Başkası kullanacaksa cevapları kendi bilgileriyle değiştirmeli.
