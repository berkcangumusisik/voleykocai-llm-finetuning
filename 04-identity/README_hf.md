---
language:
- tr
- en
license: mit
task_categories:
- text-generation
tags:
- identity
- identity-finetuning
- volleyball
- voleybol
- turkish
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: turkish
    path: turkish.jsonl
  - split: english
    path: english.jsonl
---

# VoleykoçAI — Kimlik Eğitimi Veri Seti

Bir dil modeline **kim olduğunu** öğretmek için hazırlanmış veri seti: adı, yaratıcısı, görevi, yetenekleri ve sınırları. Yapay zekâ dersinin ek ödevi kapsamında hazırlandı.

Yapı, [`alibayram/identity_finetune_magibu_q3`](https://huggingface.co/datasets/alibayram/identity_finetune_magibu_q3) veri seti referans alınarak kuruldu — aynı `turkish` / `english` bölümlemesi ve aynı satır şeması.

Eğitilen model: [`berkcangumusisik/voleykoc-identity-lora`](https://huggingface.co/berkcangumusisik/voleykoc-identity-lora).

## Bölümler

| Bölüm | Örnek |
|---|---:|
| `turkish` | 105 |
| `english` | 77 |
| **Toplam** | **182** |

## Şema

```json
{
  "system": "Sen VoleykoçAI'sın. Berkcan Gümüşışık tarafından geliştirilmiş, Türkçe konuşan bir voleybol antrenörlük asistanısın.",
  "source": "identity-yaratici",
  "conversations": [
    {"role": "user", "content": "Seni kim yaptı?"},
    {"role": "assistant", "content": "Beni Berkcan Gümüşışık geliştirdi. Bir yapay zekâ dersi kapsamında..."}
  ],
  "num_turns": 2
}
```

`source` alanı kimlik kategorisini taşır:

| Kategori | Neyi öğretir |
|---|---|
| `identity-isim` | Adı: VoleykoçAI |
| `identity-yaratici` | Yaratıcısı: Berkcan Gümüşışık |
| `identity-gorev` | Ne için var |
| `identity-yetenek` | Neler yapabilir |
| `identity-sinir` | Neyi yapmaz (teşhis koymaz, alan dışına çıkmaz) |
| `identity-dil` | Hangi dilde konuşur |
| `identity-teknik` | Nasıl eğitildi |

## Nasıl üretildi

26 kimlik soru-cevabı (15 Türkçe, 11 İngilizce) elle yazıldı, ardından her biri soru kalıplarıyla çoğaltıldı.

Kimlik verisinde aynı cevabın birden çok soru biçimine bağlanması **kasıtlıdır**: "adın ne", "kimsin", "kendini tanıt" hepsi aynı gerçeğe çıkar ve model kimliğini bu tekrardan öğrenir.

`sinir` kategorisi bilinçli olarak dahil edildi — bir kimlik yalnızca modelin ne olduğu değil, ne *olmadığı* ile de tanımlanır. Bu kategorideki örnekler modele sağlık teşhisi koymamayı ve alan dışı sorularda kullanıcıyı doğru yere yönlendirmeyi öğretir.

Üretim kodu: [github.com/berkcangumusisik/voleykocai-llm-finetuning](https://github.com/berkcangumusisik/voleykocai-llm-finetuning) → `04-identity/`

## Kullanım

```python
from datasets import load_dataset, concatenate_datasets

tr = load_dataset("berkcangumusisik/voleykoc-identity-tr", split="turkish")
en = load_dataset("berkcangumusisik/voleykoc-identity-tr", split="english")
ds = concatenate_datasets([tr, en])
```

## Sınırlar

- Küçük ve kasıtlı olarak tekrarlı; yalnızca kimlik öğretmek için tasarlandı.
- Uzun eğitimde modelin genel yeteneğini bozabilir (catastrophic forgetting). Referans notebook `max_steps=60` ile sınırlı tutar.
- Bu veri seti başka bir kimliği (`VoleykoçAI` / Berkcan Gümüşışık) taşır; kendi modelinde kullanacaksan cevapları kendi bilgilerinle değiştirmelisin.

## Lisans

MIT.
