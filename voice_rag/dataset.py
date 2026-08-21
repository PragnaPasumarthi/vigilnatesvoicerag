"""
MSMARCO-XI dataset loader.

Loads passages, queries, and ground-truth answers from the HuggingFace
ai4bharat/MSMARCO-XI dataset. Each example contains:
  - English passages (10 per query) with is_selected ground truth
  - English queries and answers
  - Query type (DESCRIPTION, ENTITY, NUMERIC, PERSON, LOCATION)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Passage:
    """A single passage with its metadata."""
    text: str
    passage_id: int
    query_id: int
    is_relevant: bool  # ground truth from is_selected
    strategy: str = "msmarco_raw"  # original passage, not chunked yet


@dataclass
class QueryExample:
    """A single query with its passages and ground truth."""
    query_id: int
    query: str  # English query
    answer: str  # English answer
    query_type: str  # DESCRIPTION, ENTITY, NUMERIC, PERSON, LOCATION
    passages: list[Passage] = field(default_factory=list)

    @property
    def relevant_passages(self) -> list[Passage]:
        return [p for p in self.passages if p.is_relevant]

    @property
    def irrelevant_passages(self) -> list[Passage]:
        return [p for p in self.passages if not p.is_relevant]


@dataclass
class MSMARCODataset:
    """Full dataset with queries, passages, and index."""
    examples: list[QueryExample] = field(default_factory=list)
    _passage_index: dict[str, Passage] = field(default_factory=dict, repr=False)

    @property
    def num_queries(self) -> int:
        return len(self.examples)

    @property
    def num_unique_passages(self) -> int:
        return len(self._passage_index)

    @property
    def queries_by_type(self) -> dict[str, list[QueryExample]]:
        result: dict[str, list[QueryExample]] = {}
        for ex in self.examples:
            result.setdefault(ex.query_type, []).append(ex)
        return result

    def get_all_passage_texts(self) -> list[str]:
        """Get all unique passage texts."""
        return list(self._passage_index.keys())


def load_msmarco_xi(
    data_path: str = "data/msmarco_xi_sample.json",
    max_examples: Optional[int] = None,
) -> MSMARCODataset:
    """
    Load the MSMARCO-XI dataset from a JSON file.

    Args:
        data_path: Path to the JSON data file
        max_examples: Maximum number of examples to load

    Returns:
        MSMARCODataset with all queries and passages
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_examples:
        raw_data = raw_data[:max_examples]

    dataset = MSMARCODataset()

    for ex in raw_data:
        passages = []
        eng_passages = ex["passages"]["English_passages"]
        is_selected = ex["passages"]["is_selected"]

        for j, (passage_text, selected) in enumerate(zip(eng_passages, is_selected)):
            if not passage_text or not passage_text.strip():
                continue

            p = Passage(
                text=passage_text.strip(),
                passage_id=j,
                query_id=ex["query_id"],
                is_relevant=bool(selected),
            )
            passages.append(p)
            # Index by text for dedup
            dataset._passage_index[passage_text.strip()] = p

        qe = QueryExample(
            query_id=ex["query_id"],
            query=ex["Eng_Query"].strip().lstrip(".").strip(),
            answer=ex["Eng_Answer"].strip(),
            query_type=ex["query_type"],
            passages=passages,
        )
        dataset.examples.append(qe)

    return dataset


def load_msmarco_xi_from_hub(
    max_examples: int = 200,
) -> MSMARCODataset:
    """
    Load directly from HuggingFace Hub (downloads if needed).

    Args:
        max_examples: How many examples to stream

    Returns:
        MSMARCODataset
    """
    from datasets import load_dataset

    ds = load_dataset("ai4bharat/MSMARCO-XI", split="validation", streaming=True)
    dataset = MSMARCODataset()

    for i, ex in enumerate(ds):
        if i >= max_examples:
            break

        passages = []
        eng_passages = ex["passages"]["English_passages"]
        is_selected = ex["passages"]["is_selected"]

        for j, (passage_text, selected) in enumerate(zip(eng_passages, is_selected)):
            if not passage_text or not passage_text.strip():
                continue

            p = Passage(
                text=passage_text.strip(),
                passage_id=j,
                query_id=ex["query_id"],
                is_relevant=bool(selected),
            )
            passages.append(p)
            dataset._passage_index[passage_text.strip()] = p

        qe = QueryExample(
            query_id=ex["query_id"],
            query=ex["Eng_Query"].strip().lstrip(".").strip(),
            answer=ex["Eng_Answer"].strip(),
            query_type=ex["query_type"],
            passages=passages,
        )
        dataset.examples.append(qe)

    return dataset
