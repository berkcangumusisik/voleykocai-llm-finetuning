#!/usr/bin/env python3
"""Replaces the auto-generated model card on the assignment 3 adapter repo.

The adapter itself is pushed from the Colab notebook; Unsloth writes a generic
README next to it. This uploads a proper card over that one and touches
nothing else in the repo.

Run:
    python 03-finetune/upload_card.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_upload import confirm_and_upload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    confirm_and_upload(
        repo_name="voleykoc-qwen3-4b-lora",
        repo_type="model",
        files=[(os.path.join(HERE, "README_hf.md"), "README.md")],
    )


if __name__ == "__main__":
    main()
