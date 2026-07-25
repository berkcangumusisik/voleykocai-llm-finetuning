#!/usr/bin/env python3
"""Publishes the VoleykoçAI domain benchmark as a Hugging Face dataset.

Requires `hf auth login` first. Prints what it will upload and asks for
confirmation.

Run:
    python 05-benchmark/upload_benchmark.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_upload import confirm_and_upload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    bench = os.path.join(ROOT, "data", "benchmark", "voleykoc_benchmark.jsonl")
    if not os.path.exists(bench):
        print("Benchmark yok. Önce: python 05-benchmark/build_benchmark.py")
        sys.exit(1)

    confirm_and_upload(
        repo_name="voleykoc-benchmark",
        repo_type="dataset",
        files=[
            (bench, "voleykoc_benchmark.jsonl"),
            (os.path.join(HERE, "README_hf.md"), "README.md"),
        ],
    )


if __name__ == "__main__":
    main()
