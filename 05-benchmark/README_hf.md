---
language:
- tr
license: mit
task_categories:
- question-answering
- multiple-choice
tags:
- volleyball
- voleybol
- turkish
- benchmark
- evaluation
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: voleykoc_benchmark.jsonl
---

# VoleykoçAI Alan Benchmark: Türkçe Voleybol

[`berkcangumusisik/voleykoc-qwen3-4b-lora`](https://huggingface.co/berkcangumusisik/voleykoc-qwen3-4b-lora) modelinin kendi alanında (Türkçe voleybol antrenörlüğü) değerlendirilmesi için elle yazılmış çoktan seçmeli test seti. Bir yapay zekâ dersi ödevi kapsamında hazırlandı.

102 soru, çok sayıda konu (kural, teknik, taktik, libero, kondisyon, sakatlık, antrenman, genel). Her soru tek doğru cevaplı.

## Neden bu benchmark?

Genel MMLU (`alibayram/yapay_zeka_turkce_mmlu`) modelin genel kültürünü ölçüyor, ama modelin uzmanlaştığı dar alanı (voleybol) ölçemez. Bu test o boşluğu kapatır: base model ile fine-tune edilmiş modeli **kendi alanlarında** karşılaştırır.

## Held-out garantisi

Tüm sorular bu test seti için ayrıca yazıldı ve modelin eğitim verisinde ([`voleykoc-antrenorluk-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-antrenorluk-tr)) yer almıyor. Yani model bu soruları eğitimde görmedi; skor gerçek genellemeyi ölçer, ezberi değil.

## Şema

```json
{
  "soru": "5-1 rotasyon sisteminde takımda kaç pasör vardır?",
  "secenekler": ["1", "2", "3", "Pasör yoktur"],
  "cevap": 0,
  "konu": "taktik"
}
```

| Alan | Açıklama |
|---|---|
| `soru` | Soru metni |
| `secenekler` | Şık listesi (A, B, C... sırasıyla) |
| `cevap` | Doğru şıkkın 0-tabanlı indeksi |
| `konu` | Konu etiketi |

## Puanlama

`alibayram/.../olcum.py` ile aynı harf-eşleşme mantığı: modelin verdiği harf doğru şıkkın harfiyle karşılaştırılır. Çoktan seçmeli olduğu için deterministik ve tekrar edilebilir.

## Kullanım

```python
from datasets import load_dataset
ds = load_dataset("berkcangumusisik/voleykoc-benchmark", split="train")
```

## Sınır

102 soru orta ölçekli bir settir; bölüm bazlı farklar gürültü içerir. Çoktan seçmeli format, prose (açık uçlu) cevap veren bir modelin tüm yeteneğini yansıtmayabilir; bu benchmark bilgi doğruluğunu ölçer, üslubu değil.

## Lisans

MIT.
