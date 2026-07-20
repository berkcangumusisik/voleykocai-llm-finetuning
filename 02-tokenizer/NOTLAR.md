# 02-tokenizer: notlar

Ödev 2: alana özel BPE tokenizer.

## Çalıştırma sırası

```bash
python 02-tokenizer/train_tokenizer.py   # 01-dataset çıktılarına bağlı
python 02-tokenizer/upload.py            # hf auth login gerekir
```

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `bpe.py` | BPE'nin sıfırdan, saf stdlib implementasyonu. **Yayımlanan tokenizer bu değil**: referans/kanıt olarak duruyor. Önceki ödevimden (countries-bpe-tokenizer) geldi, o yüzden yorumları İngilizce. |
| `train_tokenizer.py` | `tokenizers` kütüphanesiyle byte-level BPE eğitir, HF formatında kaydeder |
| `voleykoc-bpe-tokenizer/` | Eğitim çıktısı: HF'ye yüklenen dosyalar |
| `upload.py` | HF'ye yükler (onay sorar) |
| `README_hf.md` | HF model kartı |

## Neden iki BPE var?

`bpe.py` kendi JSON formatını yazıyor (`{"vocab_size": ..., "merges": [...]}`), Hugging Face onu okuyamıyor. Yayımlanabilir tokenizer için `tokenizers` kütüphanesi gerekiyordu. Algoritma ikisinde de aynı: byte-level BPE, en sık ardışık çifti birleştir.

## Bilmem gerekenler

- **Sözlük hedefi 16.000 ama gerçekleşen 7.532.** Korpus küçük ve `min_frequency=2` uygulanıyor, birleştirmeler erken tükeniyor. Bu bir hata değil; sözlüğü şişirmek yerine gerçek sayıyı bıraktım.
- **Sistem mesajı korpusa bir kez giriyor.** Veri setinin her satırında aynı sistem mesajı var; hepsini yazsaydım tokenizer o cümleyi ezberlerdi.
- **`upload.py` klasörde ne varsa onu yükler.** `save_pretrained()` transformers sürümüne göre farklı dosya kümesi yazıyor (5.x `special_tokens_map.json` üretmiyor), sabit liste yazsam sürüm değişince kırılırdı.
- **Qwen karşılaştırması internet ister.** Bağlantı yoksa o bölüm rapordan sessizce düşer, script yine de tamamlanır.
- Özel tokenlar (`<|im_start|>` vb.) Qwen uyumlu seçildi ki aynı sohbet şablonuyla yan yana denenebilsin.
