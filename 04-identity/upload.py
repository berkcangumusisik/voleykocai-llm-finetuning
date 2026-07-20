#!/usr/bin/env python3
"""Ek ödev kimlik veri setini Hugging Face'e yükler.

İki bölüm yükleniyor: turkish ve english. Hocanın
alibayram/identity_finetune_magibu_q3 veri setindeki bölümleme aynen korunuyor.

Run:
    python 04-identity/upload.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_upload import confirm_and_upload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY_DIR = os.path.join(ROOT, "data", "identity")


def main() -> None:
    confirm_and_upload(
        repo_name="voleykoc-identity-tr",
        repo_type="dataset",
        files=[
            (os.path.join(IDENTITY_DIR, "turkish.jsonl"), "turkish.jsonl"),
            (os.path.join(IDENTITY_DIR, "english.jsonl"), "english.jsonl"),
            (os.path.join(HERE, "README_hf.md"), "README.md"),
        ],
    )


if __name__ == "__main__":
    main()
