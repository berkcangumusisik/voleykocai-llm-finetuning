#!/usr/bin/env python3
"""Ödev 1 veri setini Hugging Face'e yükler.

Çalıştırmadan önce `hf auth login` yapmış olman gerekiyor. Script neyi nereye
yükleyeceğini yazdırıp onay bekler.

Run:
    python 01-dataset/upload.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_upload import confirm_and_upload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    confirm_and_upload(
        repo_name="voleykoc-antrenorluk-tr",
        repo_type="dataset",
        files=[
            (os.path.join(ROOT, "data", "train.jsonl"), "train.jsonl"),
            (os.path.join(HERE, "README_hf.md"), "README.md"),
        ],
    )


if __name__ == "__main__":
    main()
