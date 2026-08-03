#!/usr/bin/env python3
"""Verify code integrity and split generation without fitting any model."""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

import build_nested_search_splits as split_builder


CODE_DIR = Path(__file__).resolve().parent
PUBLIC_CODE_DIR = CODE_DIR.parent
PACKAGE_ROOT = CODE_DIR.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--skip-split-smoke", action="store_true")
    args = parser.parse_args()

    provenance = json.loads(
        (CODE_DIR / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8")
    )
    mismatches: list[str] = []
    for item in provenance["files"]:
        path = CODE_DIR / item["copy"]
        if not path.exists() or sha256(path) != item["copy_sha256"]:
            mismatches.append(item["copy"])
    if mismatches:
        raise RuntimeError("Code provenance mismatch: " + ", ".join(mismatches))

    python_files = sorted(PUBLIC_CODE_DIR.glob("*.py")) + sorted(
        CODE_DIR.glob("*.py")
    )
    with tempfile.TemporaryDirectory(prefix="lgbm_variance_compile_") as directory:
        compile_dir = Path(directory)
        for number, path in enumerate(python_files):
            py_compile.compile(
                str(path),
                cfile=str(compile_dir / f"{number:02d}.pyc"),
                doraise=True,
            )
    print(f"[pass] {len(python_files)} Python files compile")
    print(f"[pass] {len(provenance['files'])} preserved source hashes match")

    if not args.skip_split_smoke:
        package_root = args.package_root.expanduser().resolve()
        source = package_root / "Input" / "splits_search24.xlsx"
        if not source.exists():
            raise FileNotFoundError(
                f"Split smoke requires {source}; use --skip-split-smoke if inputs are not copied yet."
            )
        with tempfile.TemporaryDirectory(prefix="lgbm_variance_split_") as directory:
            root = Path(directory)
            (root / "Input").mkdir()
            shutil.copy2(source, root / "Input" / source.name)
            split_builder.build_nested_splits(root, backend="openpyxl")
            search36 = pd.read_excel(
                root / "Input" / "splits_search36.xlsx", sheet_name="24"
            )
            search48 = pd.read_excel(
                root / "Input" / "splits_search48.xlsx", sheet_name="24"
            )
            if len(search36) != 36 or len(search48) != 48:
                raise RuntimeError("Portable split writer returned incorrect row counts")
            if list(search36.columns) != list(split_builder.COLUMNS):
                raise RuntimeError("SEARCH-36 columns differ from the registered design")
            if list(search48.columns) != list(split_builder.COLUMNS):
                raise RuntimeError("SEARCH-48 columns differ from the registered design")
        print("[pass] portable nested split writer produced 36 and 48 rows")

    print("Setup verification passed. No model was fitted or evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
