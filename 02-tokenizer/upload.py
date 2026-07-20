#!/usr/bin/env python3
"""Ödev 2 BPE tokenizer'ını Hugging Face'e yükler.

Çalıştırmadan önce `hf auth login` yapmış olman ve train_tokenizer.py'yi
çalıştırmış olman gerekiyor.

Run:
    python 02-tokenizer/upload.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_upload import confirm_and_upload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.path.join(HERE, "voleykoc-bpe-tokenizer")

# save_pretrained() transformers sürümüne göre farklı dosya kümesi yazıyor
# (5.x special_tokens_map.json üretmiyor), o yüzden klasörde ne varsa onu
# yüklüyorum -- sabit liste yazsam sürüm değişince kırılırdı.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]


def main() -> None:
    files = [
        (os.path.join(TOK_DIR, name), name)
        for name in TOKENIZER_FILES
        if os.path.exists(os.path.join(TOK_DIR, name))
    ]
    if not files:
        print(f"{os.path.relpath(TOK_DIR, ROOT)} boş. "
              f"Önce: python 02-tokenizer/train_tokenizer.py")
        sys.exit(1)

    files.append((os.path.join(HERE, "README_hf.md"), "README.md"))

    confirm_and_upload(
        repo_name="voleykoc-bpe-tokenizer",
        repo_type="model",
        files=files,
    )


if __name__ == "__main__":
    main()
