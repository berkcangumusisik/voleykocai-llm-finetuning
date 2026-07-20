#!/usr/bin/env python3
"""Uploads the assignment 1 dataset to Hugging Face.

Requires `hf auth login` first. Prints what it will upload and waits for
confirmation.

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
