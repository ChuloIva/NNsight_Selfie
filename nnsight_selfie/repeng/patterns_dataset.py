"""
Utilities for loading and pairing cognitive pattern examples from positive_patterns.jsonl.

This module builds RePENG-style `DatasetEntry` pairs from the project's
`data/final/positive_patterns.jsonl` file, supporting pairings like:
- positive vs negative
- positive vs transition
- negative vs transition

Each pairing returns a list of `DatasetEntry` objects in the alternating order
expected by the RePENG activation extractor.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import json

from .repeng_dataset_generator import DatasetEntry


PairType = Literal["pos-neg", "pos-trans", "neg-trans"]


@dataclass
class PatternRecord:
    pattern_name: str
    positive: str
    negative: str
    transition: str


def load_positive_patterns_jsonl(path: str) -> Dict[str, List[PatternRecord]]:
    """
    Load the project's positive_patterns.jsonl into a dict keyed by pattern name.

    Expects each JSON line to contain at least:
    - positive_thought_pattern (str)
    - reference_negative_example (str)
    - reference_transformed_example (str)
    - cognitive_pattern_name (str)

    Returns:
        Dict[pattern_name, List[PatternRecord]]
    """
    pattern_to_records: Dict[str, List[PatternRecord]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            pos = obj.get("positive_thought_pattern", "").strip()
            neg = obj.get("reference_negative_example", "").strip()
            trans = obj.get("reference_transformed_example", "").strip()
            name = obj.get("cognitive_pattern_name", "unknown").strip() or "unknown"

            if not pos or not neg or not trans:
                continue

            rec = PatternRecord(pattern_name=name, positive=pos, negative=neg, transition=trans)
            pattern_to_records.setdefault(name, []).append(rec)

    return pattern_to_records


def list_patterns(path: str) -> List[str]:
    """Return sorted list of pattern names in the JSONL file."""
    records = load_positive_patterns_jsonl(path)
    return sorted(records.keys())


def build_dataset_for_pair(records: List[PatternRecord], pair_type: PairType) -> List[DatasetEntry]:
    """
    Build a list of DatasetEntry pairs for a specific pattern and pairing type.

    Args:
        records: List of PatternRecord for one cognitive pattern
        pair_type: One of 'pos-neg', 'pos-trans', 'neg-trans'

    Returns:
        List[DatasetEntry] in alternating order internally expected by extractor.
    """
    dataset: List[DatasetEntry] = []

    for rec in records:
        if pair_type == "pos-neg":
            dataset.append(DatasetEntry(positive=rec.positive, negative=rec.negative))
        elif pair_type == "pos-trans":
            dataset.append(DatasetEntry(positive=rec.positive, negative=rec.transition))
        elif pair_type == "neg-trans":
            dataset.append(DatasetEntry(positive=rec.negative, negative=rec.transition))
        else:
            raise ValueError(f"Unknown pair_type: {pair_type}")

    return dataset


def build_all_datasets(
    path: str,
    pair_types: Optional[List[PairType]] = None,
    max_patterns: Optional[int] = None
) -> Dict[str, Dict[PairType, List[DatasetEntry]]]:
    """
    Build datasets for all patterns and requested pairings.

    Args:
        path: Path to patterns JSONL file
        pair_types: Which pair types to build
        max_patterns: Maximum number of patterns to load (for memory optimization)

    Returns:
        Dict mapping pattern_name -> { pair_type -> dataset entries }
    """
    if pair_types is None:
        pair_types = ["pos-neg", "pos-trans", "neg-trans"]

    pattern_to_records = load_positive_patterns_jsonl(path)
    out: Dict[str, Dict[PairType, List[DatasetEntry]]] = {}

    # Limit patterns if specified
    pattern_items = list(pattern_to_records.items())
    if max_patterns is not None:
        pattern_items = pattern_items[:max_patterns]
        print(f"Limited to {len(pattern_items)} patterns for memory optimization")

    for pattern_name, records in pattern_items:
        out[pattern_name] = {}
        for pt in pair_types:
            out[pattern_name][pt] = build_dataset_for_pair(records, pt)

    return out


