# 03-finetune: notlar

Ödev 3: Unsloth ile LoRA fine-tune.

## Bu klasör lokalde çalışmaz

Unsloth Triton'a dayanıyor, Triton macOS'ta yok. Bu depo Apple Silicon üzerinde geliştirildi, dolayısıyla eğitim **Google Colab'da** yapılıyor.

## Çalıştırma

1. `finetune_voleykoc.ipynb` dosyasını https://colab.research.google.com adresine yükle
2. `Runtime → Change runtime type → T4 GPU`
3. Soldaki 🔑 **Secrets** panelinden `HF_TOKEN` ekle (Write yetkili), **Notebook access** anahtarını aç
4. Hücreleri sırayla çalıştır

**Ön koşul:** veri seti HF'de yayımlanmış olmalı: notebook onu Hub'dan çekiyor. Yani önce `01-dataset/upload.py`.

## Reçete

| | |
|---|---|
| Temel model | `unsloth/Qwen3-4B-Instruct-2507` |
| Quantization | 4-bit |
| LoRA | r=16, alpha=16, dropout=0 |
| Hedef katmanlar | q, k, v, o, gate, up, down |
| max_seq_length | 2048 |
| Eğitim | 3 epoch, lr=2e-4, batch 2 × accum 4 |
| Seed | 1337 |

## Bilmem gerekenler

- **Notebook eğitim öncesi ve sonrası aynı 5 soruyu soruyor.** Cevaplar `karsilastirma.json`'a yazılıyor; indirip `reports/` altına koy. Fine-tune'un işe yaradığını gösteren asıl kanıt bu, kayıp eğrisi değil.
- **Sadece LoRA adaptörü yükleniyor** (~100 MB), birleştirilmiş model değil (~8 GB).
- **Bu notebook Ödev 2'deki tokenizer'ı kullanmıyor.** Temel modelin kendi tokenizer'ıyla eğitiliyor. Sebebi README'de "Neden tokenizer fine-tune'da kullanılmadı?" başlığında.
- **Token'ı hücreye düz metin yazma.** Notebook GitHub'a commit'leniyor.
- `OutOfMemoryError` alırsan: `MAX_SEQ_LENGTH` 2048 → 1024, ya da temel modeli `unsloth/Qwen3-1.7B-Instruct` yap.
