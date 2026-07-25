"""
Hazırlanan JSONL veri setini Hugging Face Hub'daki profiline yükler.

Kullanım:
    pip install datasets huggingface_hub
    huggingface-cli login   # ya da: export HF_TOKEN=...
    python push_dataset_to_hub.py
"""

import os
from datasets import load_dataset

# Öncelik sırası: augment edilmiş dosya varsa onu, yoksa seed dosyasını kullan
CANDIDATES = [
    "tarihi_yerler_augmented.jsonl",
    "tarihi_yerler_dataset.jsonl",
    "tarihi_yerler_seed.jsonl",
]

HF_REPO_ID = "KULLANICI_ADIN/tarihi-yerler-tr-dataset"  # <-- kendi HF kullanıcı adınla değiştir

data_file = next((f for f in CANDIDATES if os.path.exists(os.path.join(os.path.dirname(__file__), f))), None)
if data_file is None:
    raise FileNotFoundError("Önce build_dataset.py / augment_with_llm.py çalıştırılmalı.")

path = os.path.join(os.path.dirname(__file__), data_file)
print(f"Yüklenen dosya: {path}")

ds = load_dataset("json", data_files=path, split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)

print(ds)
ds.push_to_hub(HF_REPO_ID, private=False)
print(f"Veri seti yüklendi: https://huggingface.co/datasets/{HF_REPO_ID}")
