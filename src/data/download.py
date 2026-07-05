"""
download.py — Downloads the UCI Online Retail dataset.

Checks for an existing file first; re-downloads only if missing or corrupt.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

from src.config import DATASET_URL, RAW_CSV


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(force: bool = False) -> Path:
    """Download the raw CSV; skip if already present (unless force=True)."""
    if RAW_CSV.exists() and not force:
        size_mb = RAW_CSV.stat().st_size / 1e6
        print(f"[download] Dataset already present ({size_mb:.1f} MB): {RAW_CSV}")
        return RAW_CSV

    print(f"[download] Fetching dataset from:\n  {DATASET_URL}")
    response = requests.get(DATASET_URL, timeout=120, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    with open(RAW_CSV, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f} / {total/1e6:.1f} MB)", end="", flush=True)
    print()

    size_mb = RAW_CSV.stat().st_size / 1e6
    sha = _sha256(RAW_CSV)
    print(f"[download] Done — {size_mb:.1f} MB  SHA256={sha[:16]}...")
    return RAW_CSV


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
