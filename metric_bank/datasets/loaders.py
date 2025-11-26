"""Dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Union

from ..core.types import Example


def load_examples(
    dataset: Union[str, Path, Sequence[Example], Sequence[dict]]
) -> List[Example]:
    """Normalize several dataset formats into Example objects."""
    if isinstance(dataset, Example):
        return [dataset]
    if isinstance(dataset, (str, Path)):
        records = _load_from_path(Path(dataset))
    elif isinstance(dataset, Sequence):
        if dataset and isinstance(dataset[0], Example):
            return list(dataset)
        records = list(dataset)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")
    return [_dict_to_example(idx, record) for idx, record in enumerate(records)]


def _load_from_path(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        return data
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _dict_to_example(idx: int, record: dict) -> Example:
    inputs = record.get("inputs")
    if inputs is None:
        # allow simple shape {"question": "...", "reference": "..."}
        candidate_inputs = {
            key: record[key]
            for key in ["question", "prompt", "query"]
            if key in record
        }
        inputs = candidate_inputs or {}
    return Example(
        id=str(record.get("id") or f"dataset_{idx}"),
        inputs=inputs,
        output=record.get("output"),
        reference=record.get("reference"),
        context=record.get("context"),
        metadata=record.get("metadata") or {},
    )

