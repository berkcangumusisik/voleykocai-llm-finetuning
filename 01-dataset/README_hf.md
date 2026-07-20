---
language:
- tr
license: mit
task_categories:
- text-generation
- question-answering
tags:
- volleyball
- voleybol
- turkish
- sports
- coaching
- instruction-tuning
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# VoleykoçAI — Türkçe Voleybol Antrenörlüğü Veri Seti

Türkçe voleybol antrenörlüğü alanında soru-cevap veri seti. Teknik (manşet, parmak pas, smaç, servis, blok), taktik ve rotasyon sistemleri, antrenman planlaması, kondisyon, sakatlık önleme ve oyun kuralları konularını kapsıyor.

Bir yapay zekâ dersi ödevi kapsamında hazırlandı. Eğitilen model: [`berkcangumusisik/voleykoc-qwen3-4b-lora`](https://huggingface.co/berkcangumusisik/voleykoc-qwen3-4b-lora).

## Şema

Şema, [`alibayram/identity_finetune_magibu_q3`](https://huggingface.co/datasets/alibayram/identity_finetune_magibu_q3) veri setindeki magibu düzenini izler:

```json
{
  "system": "Sen VoleykoçAI'sın: Türkçe konuşan bir voleybol antrenörlük asistanısın...",
  "source": "wikipedia",
  "conversations": [
    {"role": "user", "content": "5-1 rotasyon sistemi nasıl çalışır?"},
    {"role": "assistant", "content": "5-1'de takımda tek pasör vardır ve altı rotasyonun tamamında..."}
  ],
  "num_turns": 2
}
```

| Alan | Tip | Açıklama |
|---|---|---|
| `system` | string | Sabit sistem mesajı; asistanın rolünü tanımlar |
| `source` | string | `wikipedia`, `tvf` veya `synthetic` |
| `conversations` | list | `{role, content}` mesajları, sırayla user → assistant |
| `num_turns` | int | Mesaj sayısı (bu veri setinde hep 2) |

## Nasıl üretildi

Üç kaynaktan derlendi:

1. **Web scraping — Türkçe Wikipedia.** Voleybol, Plaj voleybolu, Oturarak voleybol, FIVB, CEV, TVF, millî takımlar, Sultanlar/Efeler Ligi ve büyük kulüplerin sayfaları. Her bölüm başlığı bir soruya, paragrafları cevaba dönüştürüldü.
2. **Web scraping — Türkiye Voleybol Federasyonu.** `tvf.org.tr` haber sayfaları. (Federasyonun oyun kuralları kitapçığı sitede yalnızca PDF olarak yayımlandığı için haber akışı kullanıldı.)
3. **Sentetik.** 20 adet elle yazılmış gerçek antrenörlük soru-cevabı tohum olarak alındı ve bir dil modeliyle çoğaltıldı.

Birleştirme sonrası normalize edilmiş soru üzerinden tekilleştirme, uzunluk filtresi ve sabit tohumla (`seed=1337`) karıştırma uygulandı.

Üretim kodunun tamamı: [github.com/berkcangumusisik/voleykocai-llm-finetuning](https://github.com/berkcangumusisik/voleykocai-llm-finetuning)

## Kullanım

```python
from datasets import load_dataset

ds = load_dataset("berkcangumusisik/voleykoc-antrenorluk-tr", split="train")
print(ds[0]["conversations"][0]["content"])
```

## Sınırlar

- Veri seti küçük; bir alan adaptasyonu denemesi için tasarlandı, kapsamlı bir voleybol bilgi tabanı değil.
- Wikipedia kaynaklı cevaplar ansiklopedik dille yazılmıştır, antrenör diliyle değil.
- **Sağlık tavsiyesi değildir.** Sakatlık ve tedaviyle ilgili içerik yalnızca genel bilgi amaçlıdır; teşhis ve tedavi için spor hekimine başvurun.

## Lisans ve atıflar

Veri seti MIT lisansıyla paylaşılıyor. Kaynakların kendi lisansları geçerlidir:

- Türkçe Wikipedia içeriği **CC BY-SA 4.0** altındadır. Türetilmiş metinler bu lisansa tabidir.
- Türkiye Voleybol Federasyonu içeriği tvf.org.tr'ye aittir; burada yalnızca eğitim amaçlı, dönüştürülmüş biçimde kullanılmıştır.

## Alıntı

```bibtex
@misc{voleykocai2026dataset,
  author = {Gümüşışık, Berkcan},
  title  = {VoleykoçAI: Türkçe Voleybol Antrenörlüğü Veri Seti},
  year   = {2026},
  url    = {https://huggingface.co/datasets/berkcangumusisik/voleykoc-antrenorluk-tr}
}
```
