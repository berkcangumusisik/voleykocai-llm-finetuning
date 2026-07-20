---
language:
- tr
license: mit
library_name: peft
base_model: unsloth/Qwen3-4B-Instruct-2507
tags:
- lora
- peft
- unsloth
- qwen3
- volleyball
- voleybol
- turkish
- coaching
datasets:
- berkcangumusisik/voleykoc-antrenorluk-tr
pipeline_tag: text-generation
---

# VoleykoçAI: Türkçe Voleybol Antrenörlüğü LoRA Adaptörü

Qwen3-4B-Instruct-2507 üzerine, Türkçe voleybol antrenörlüğü verisiyle eğitilmiş LoRA adaptörü. Bir yapay zekâ dersi ödevi kapsamında hazırlandı.

Teknik, taktik ve rotasyon sistemleri, antrenman planlaması, kondisyon, sakatlık önleme ve oyun kuralları konularında Türkçe cevap verir.

## Eğitim

| | |
|---|---|
| Temel model | [`unsloth/Qwen3-4B-Instruct-2507`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507) |
| Yöntem | 4-bit QLoRA (Unsloth) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Hedef katmanlar | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| max_seq_length | 2048 |
| Epoch | 3 |
| Learning rate | 2e-4 |
| Efektif batch | 8 (batch 2 x accum 4) |
| Optimizer | adamw_8bit |
| Seed | 1337 |
| Donanım | Google Colab T4 |

Veri seti: [`berkcangumusisik/voleykoc-antrenorluk-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-antrenorluk-tr), 166 örnek.

## Kullanım

Unsloth ile:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="berkcangumusisik/voleykoc-qwen3-4b-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

SYSTEM = (
    "Sen VoleykoçAI'sın: Türkçe konuşan bir voleybol antrenörlük asistanısın. "
    "Teknik, taktik, antrenman planlaması, kondisyon ve oyun kuralları "
    "konularında somut ve uygulanabilir cevaplar verirsin."
)

mesajlar = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "5-1 rotasyon sistemi nasıl çalışır?"},
]
girdi = tokenizer.apply_chat_template(
    mesajlar, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
cikti = model.generate(input_ids=girdi, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tokenizer.decode(cikti[0][girdi.shape[1]:], skip_special_tokens=True))
```

PEFT ile:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "berkcangumusisik/voleykoc-qwen3-4b-lora")
tokenizer = AutoTokenizer.from_pretrained("berkcangumusisik/voleykoc-qwen3-4b-lora")
```

## Sınırlar

- **Küçük veri seti.** 166 örnekle eğitildi; bu bir alan adaptasyonu denemesidir, kapsamlı bir voleybol uzmanı değil. Bilmediği konularda uydurabilir, cevapları doğrulayın.
- **Sağlık tavsiyesi değildir.** Sakatlık ve tedaviyle ilgili çıktılar genel bilgi amaçlıdır. Teşhis ve tedavi için spor hekimine başvurun.
- **Kimlik eğitimi bu adaptörde yok.** Modelin kendini VoleykoçAI olarak tanıtması ayrı bir çalışmadır: [`berkcangumusisik/voleykoc-identity-lora`](https://huggingface.co/berkcangumusisik/voleykoc-identity-lora).
- **Alana özel tokenizer bu adaptörde kullanılmadı.** [`berkcangumusisik/voleykoc-bpe-tokenizer`](https://huggingface.co/berkcangumusisik/voleykoc-bpe-tokenizer) bağımsız bir teslimdir; bu adaptör temel modelin kendi tokenizer'ıyla eğitildi. Sözlük değiştirmek embedding katmanının yeniden boyutlandırılmasını ve baştan eğitimi gerektirir.

## Eğitim kodu

[github.com/berkcangumusisik/voleykocai-llm-finetuning](https://github.com/berkcangumusisik/voleykocai-llm-finetuning) → `03-finetune/finetune_voleykoc.ipynb`

Notebook eğitim öncesi ve sonrası aynı beş soruyu sorup cevapları yan yana koyar.

## Lisans

Adaptör MIT lisanslıdır. Temel model Qwen3-4B-Instruct-2507 kendi lisansına tabidir. Eğitim verisindeki Wikipedia kaynaklı içerik CC BY-SA 4.0 altındadır.
