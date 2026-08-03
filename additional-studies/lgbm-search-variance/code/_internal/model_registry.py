"""Resolve retained model artifacts in original and publication layouts.

The completed experiment used long snapshot filenames.  The structured
publication package keeps shorter filenames and records the mapping in
``Models/MODEL_MANIFEST.json``.  This module lets the scientific code use
either layout without changing model metadata or numerical behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping


TARGET_CODES = {"P_Power": "P", "Q_Power": "Q"}


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _records(models_dir: Path) -> list[Mapping[str, Any]]:
    manifest_path = models_dir / "MODEL_MANIFEST.json"
    if not manifest_path.exists():
        return []
    payload = _read_json(manifest_path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid model manifest records at {manifest_path}")
    return [item for item in records if isinstance(item, dict)]


def _publication_path(models_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "models":
        return models_dir.parent / path
    return models_dir / path


def _record(
    models_dir: Path, snapshot: str, target: str
) -> Mapping[str, Any] | None:
    target_code = TARGET_CODES.get(target, target)
    matches = [
        item
        for item in _records(models_dir)
        if str(item.get("original_snapshot", "")) == snapshot
        and str(item.get("target", "")) == target_code
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Model manifest contains duplicate records for {snapshot}/{target}"
        )
    return matches[0] if matches else None


def metadata_path(
    models_dir: Path,
    snapshot: str,
    target: str,
    *,
    required: bool = True,
) -> Path:
    """Return metadata for a snapshot/target in either supported layout."""
    models_dir = Path(models_dir)
    legacy = models_dir / f"{snapshot}_LGBM_sheet1_{target}_best.meta.json"
    if legacy.exists():
        return legacy

    record = _record(models_dir, snapshot, target)
    publication = (
        _publication_path(models_dir, record.get("publication_metadata_file"))
        if record is not None
        else None
    )
    if publication is not None and publication.exists():
        return publication
    if required:
        expected = publication or legacy
        raise FileNotFoundError(
            f"Model metadata is unavailable for {snapshot}/{target}: {expected}"
        )
    return publication or legacy


def artifact_base(models_dir: Path, snapshot: str, target: str) -> Path:
    """Return the filename base used when writing a retained model pair."""
    models_dir = Path(models_dir)
    record = _record(models_dir, snapshot, target)
    if record is not None:
        model_path = _publication_path(
            models_dir, record.get("publication_model_file")
        )
        if model_path is not None:
            return model_path.with_suffix("")
        metadata = _publication_path(
            models_dir, record.get("publication_metadata_file")
        )
        if metadata is not None:
            suffix = ".meta.json"
            value = str(metadata)
            return Path(value[: -len(suffix)]) if value.endswith(suffix) else metadata
    return models_dir / f"{snapshot}_LGBM_sheet1_{target}_best"


def iter_metadata(models_dir: Path) -> Iterator[tuple[Path, Dict[str, Any]]]:
    """Yield each available metadata file once, including newly trained files."""
    models_dir = Path(models_dir)
    seen: set[Path] = set()
    for record in _records(models_dir):
        path = _publication_path(
            models_dir, record.get("publication_metadata_file")
        )
        if path is None or not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path, _read_json(path)

    for path in sorted(models_dir.glob("*_LGBM_sheet1_*_best.meta.json")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path, _read_json(path)
