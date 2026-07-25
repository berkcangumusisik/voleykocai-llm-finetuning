# 05-benchmark: notlar

Fine-tune edilmiş modelin Türkçe MMLU benchmark ölçümü.

## Bu klasör lokalde çalışmaz

Ölçüm 4B modeli GPU'da çalıştırdığı için **Google Colab'da** yapılıyor. `mmlu_benchmark.ipynb` dosyasını Colab'a yükle, T4 GPU seç, hücreleri sırayla çalıştır.

## Benchmark sıfırdan geliştirilmedi

Hocanın `olcum.py` dosyasından iki parça birebir korundu:

- `cevap_dogru_mu` fonksiyonu: harf eşleşmesi, ilk-harf eşleşmesi, sonra anlamsal benzerlik yedeği.
- Modele verilen prompt metni.

Kaynak: https://huggingface.co/datasets/alibayram/yapay_zeka_turkce_mmlu_bolum_sonuclari/blob/main/olcum.py

**Tek fark:** orijinal kod modeli Ollama ile çağırıyor. Bizim çıktı bir LoRA adaptörü, Ollama'ya doğrudan girmiyor; bu yüzden çıkarımı Unsloth/transformers ile yapıyorum. Ölçülen benchmark, sorular ve puanlama aynı; sadece modeli çalıştıran arka uç farklı.

## Ne ölçülüyor

| | |
|---|---|
| Benchmark | `alibayram/yapay_zeka_turkce_mmlu` (6200 soru, 62 bölüm, 5 şık) |
| Model 1 | `unsloth/Qwen3-4B-Instruct-2507` (base) |
| Model 2 | `berkcangumusisik/voleykoc-qwen3-4b-lora` (fine-tune) |
| Kıyas | liderlik tablosundan hazır skorlar (qwen3:14b, gemma2:9b vb.) |

## Beklenen sonuç

MMLU genel kültür ölçer; model dar bir voleybol verisiyle eğitildi. Fine-tune'un MMLU'yu yükseltmesi **beklenmez**; base ile aynı kalması ya da bir miktar düşmesi normaldir. Ödev karşılaştırmalı raporlama istiyor, fine-tune'un kazanmasını değil. Sonuç ne çıkarsa model kartında dürüstçe yazılır.

## Sonuç nereye gidiyor

Notebook `mmlu_sonuclari.json` üretir. İndirip `reports/` altına koy. Karşılaştırmalı tablo, notebook'un çıktısından kopyalanıp `03-finetune/README_hf.md` içine eklenir, sonra `python 03-finetune/upload_card.py` ile model kartı güncellenir.
