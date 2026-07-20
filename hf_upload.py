#!/usr/bin/env python3
"""Hugging Face yükleme yardımcıları.

Üç aşamanın upload.py'si de bunu kullanıyor; aynı onay ve hata mesajlarını üç
kez yazmayayım diye ortak modüle aldım. Doğrudan çalıştırılmaz.

Token buraya hiç yazılmıyor: `hf auth login` ile bir kez giriş yapıyorsun,
huggingface_hub token'ı kendi saklama alanından okuyor.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
HF_USER = "berkcangumusisik"


def _human(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB" if n_bytes < 1024**2 else f"{n_bytes / 1024**2:.1f} MB"


def _count_lines(path: str) -> int | None:
    if not path.endswith(".jsonl"):
        return None
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def whoami() -> str:
    """Giriş yapılmış mı? Yapılmışsa kullanıcı adını döndür."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import LocalTokenNotFoundError

    try:
        return HfApi().whoami()["name"]
    except LocalTokenNotFoundError:
        print("Hugging Face girişi yapılmamış.\n")
        print("  pip install -U huggingface_hub")
        print("  hf auth login        # WRITE yetkili token gerekiyor")
        print("\nToken: https://huggingface.co/settings/tokens")
        sys.exit(1)
    except Exception as exc:
        print(f"Giriş doğrulanamadı: {exc}")
        sys.exit(1)


def confirm_and_upload(
    repo_name: str,
    repo_type: str,
    files: list[tuple[str, str]],
    private: bool = False,
) -> None:
    """Ne yükleneceğini göster, onay al, sonra yükle.

    files: (yerel_yol, repodaki_yol) çiftleri.
    """
    from huggingface_hub import HfApi

    user = whoami()
    repo_id = f"{HF_USER}/{repo_name}"

    if user != HF_USER:
        print(f"UYARI: '{user}' olarak giriş yapılmış ama hedef repo "
              f"'{HF_USER}' hesabında.")
        print("Yanlış hesaba yüklemeyi önlemek için duruyorum.")
        print(f"Doğru hesapla giriş yap ya da hf_upload.py içindeki "
              f"HF_USER'ı '{user}' yap.")
        sys.exit(1)

    missing = [local for local, _ in files if not os.path.exists(local)]
    if missing:
        print("Şu dosyalar yok, önce üretilmeleri gerekiyor:")
        for m in missing:
            print(f"  - {os.path.relpath(m, ROOT)}")
        sys.exit(1)

    print(f"Yüklenecek repo : {repo_id} ({repo_type}, "
          f"{'private' if private else 'public'})")
    print("Dosyalar        :")
    for local, remote in files:
        size = _human(os.path.getsize(local))
        lines = _count_lines(local)
        extra = f", {lines} satır" if lines is not None else ""
        print(f"  - {remote:<28} ({size}{extra})")

    answer = input("\nDevam? [e/h]: ").strip().lower()
    if answer not in ("e", "evet", "y", "yes"):
        print("İptal edildi, hiçbir şey yüklenmedi.")
        sys.exit(0)

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    print(f"\nRepo hazır: {repo_id}")

    for local, remote in files:
        print(f"  yükleniyor: {remote} ...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type=repo_type,
        )

    base = "datasets/" if repo_type == "dataset" else ""
    print(f"\nBitti -> https://huggingface.co/{base}{repo_id}")
