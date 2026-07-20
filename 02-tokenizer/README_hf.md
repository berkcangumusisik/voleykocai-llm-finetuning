---
language:
- tr
license: mit
library_name: transformers
tags:
- tokenizer
- bpe
- byte-level-bpe
- volleyball
- voleybol
- turkish
---

# VoleykoçAI: Türkçe Voleybol BPE Tokenizer

Türkçe voleybol metinleri üzerinde sıfırdan eğitilmiş byte-level BPE tokenizer. Bir yapay zekâ dersi ödevi kapsamında hazırlandı.

| | |
|---|---|
| Algoritma | Byte-level BPE (`tokenizers` kütüphanesi) |
| Sözlük boyutu | 7.532 |
| Özel tokenlar | `<\|endoftext\|>`, `<\|im_start\|>`, `<\|im_end\|>`, `<pad>` |
| `<unk>` | Yok: byte fallback |
| Sohbet şablonu | ChatML (Qwen uyumlu) |

## Kullanım

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("berkcangumusisik/voleykoc-bpe-tokenizer")

ids = tok.encode("Pasör çaprazı hücumda ne yapar?", add_special_tokens=False)
print(tok.convert_ids_to_tokens(ids))
print(tok.decode(ids))
```

Sohbet şablonuyla:

```python
tok.apply_chat_template(
    [{"role": "user", "content": "5-1 rotasyonu nedir?"}],
    tokenize=False, add_generation_prompt=True,
)
# '<|im_start|>user\n5-1 rotasyonu nedir?<|im_end|>\n<|im_start|>assistant\n'
```

## Korpus

[`berkcangumusisik/voleykoc-antrenorluk-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-antrenorluk-tr) veri setinin tüm metni artı Türkçe Wikipedia ve TVF'den scrape edilen ham voleybol yazıları: **214.015 karakter / 29.532 kelime**.

## Alan verimliliği

Aynı 10 Türkçe voleybol cümlesinde Qwen3'ün tokenizer'ıyla karşılaştırma:

| | Toplam token |
|---|---:|
| VoleykoçAI | **114** |
| Qwen3-4B-Instruct-2507 | 179 |

Bu alanda **%36 daha az** token. Sözlük tamamen Türkçe voleybol metninden öğrenildiği için `pasör`, `manşet`, `rotasyon` gibi terimler tek parça kalıyor; Qwen'in çok dilli sözlüğü onları parçalara ayırıyor.

> Bu karşılaştırma **bu alandaki** verimliliği gösterir, genel kaliteyi değil. Qwen'in sözlüğü yüzlerce dili kapsıyor, bu tokenizer tek alanı.

## Tasarım notları

- **Byte fallback, `<unk>` yok.** Taban sözlük 256 byte olduğu için hiç görülmemiş karakterler bile (emoji, başka alfabeler) kayıpsız kodlanabiliyor. 10/10 test cümlesi encode→decode sonrası birebir aynı döndü.
- **Hedef 16.000, gerçekleşen 7.532.** Korpus küçük olduğu ve `min_frequency=2` uygulandığı için birleştirmeler daha erken tükendi. Sözlüğü yapay olarak şişirmek yerine gerçek sayıyı bırakmayı tercih ettim.
- **Özel tokenlar Qwen uyumlu** seçildi ki aynı sohbet şablonuyla yan yana denenebilsin.

## Önemli not

Bu tokenizer **bağımsız bir teslim**. [`berkcangumusisik/voleykoc-qwen3-4b-lora`](https://huggingface.co/berkcangumusisik/voleykoc-qwen3-4b-lora) adaptörü bunu **kullanmıyor**: o, temel modelin kendi tokenizer'ıyla eğitildi. Bir modelin sözlüğünü değiştirmek embedding katmanının yeniden boyutlandırılmasını ve baştan eğitimi gerektirir; ödev bunu istemiyordu.

## Eğitim kodu

[github.com/berkcangumusisik/voleykocai-llm-finetuning](https://github.com/berkcangumusisik/voleykocai-llm-finetuning) → `02-tokenizer/`

Depoda ayrıca `bpe.py` var: BPE'nin saf standart kütüphaneyle, sıfırdan yazılmış hâli (train/encode/decode/save/load). Burada yayımlanan dosyalar `tokenizers` kütüphanesiyle üretildi, çünkü Hugging Face formatı gerekiyordu.

## Lisans

MIT. Eğitim korpusundaki Wikipedia içeriği CC BY-SA 4.0 altındadır.
