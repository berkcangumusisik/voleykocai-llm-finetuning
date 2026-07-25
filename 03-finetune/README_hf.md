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

## Türkçe MMLU Benchmark

Model, [`alibayram/yapay_zeka_turkce_mmlu`](https://huggingface.co/datasets/alibayram/yapay_zeka_turkce_mmlu) benchmark'ında (6200 soru, 62 bölüm) ölçüldü ve base model ile karşılaştırıldı. Ölçüm kodu hocanın [`olcum.py`](https://huggingface.co/datasets/alibayram/yapay_zeka_turkce_mmlu_bolum_sonuclari/blob/main/olcum.py) dosyasındaki puanlama mantığıyla birebir aynı; çıkarım LoRA adaptörü için Unsloth ile yapıldı.

| Model | Parametre | Başarı (6200 soru) |
|---|---|---:|
| Base Qwen3-4B-Instruct-2507 | 4B | %54.42 (3374/6200) |
| **VoleykoçAI (fine-tune)** | 4B (LoRA) | **%52.40** (3249/6200) |
| qwen3:14b (liderlik, referans) | 14.8B | %71.65 |
| gemma2:9b (liderlik, referans) | 9.2B | %69.26 |

**Fine-tune farkı: -2.02 puan.** Base ile fine-tune arasında yalnızca 2 puanlık fark var. MMLU genel kültür ölçer (62 bölüm: hukuk, iktisat, din, aşçılık, ehliyet...); bu model ise 166 örneklik dar bir voleybol verisiyle eğitildi. Genel bilginin neredeyse tamamen korunmuş olması, dar alan uzmanlaşmasının modeli bozmadığını gösteriyor. Amaç kazanmak değil, ölçmek ve karşılaştırmalı raporlamaktı.

Bölüm bazında en çok değişenler (her bölüm 100 soru, bu ölçekte ±5-7 gürültü sayılır):

| Bölüm | Base | Fine-tune | Fark |
|---|---:|---:|---:|
| Sağlık Yönetimi | %53 | %60 | +7 |
| Kültürel Miras ve Turizm | %61 | %67 | +6 |
| Üniversite Giriş Temel Bilimler | %46 | %52 | +6 |
| DHBT | %43 | %30 | -13 |
| Sosyal Hizmetler | %58 | %48 | -10 |
| İktisat | %58 | %50 | -8 |

Ölçüm notu: `olcum.py`'nin puanlaması üç kademeli (tam harf eşleşmesi, ilk harf eşleşmesi, anlamsal benzerlik). İlk iki kademe birebir kullanıldı; üçüncü kademedeki anlamsal benzerlik modeli Colab'daki sürüm çakışması nedeniyle kapatıldı. Prompt "sadece harf yaz" dediği için model neredeyse her zaman düz harf ürettiğinden bu kademe pratikte çok seyrek devreye girerdi.

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
