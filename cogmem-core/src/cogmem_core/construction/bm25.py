from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .text import tokenize


@dataclass(slots=True)
class BM25Candidate:
    episode_index: int
    bm25_score: float
    temporal_prior: float
    combined_score: float
    reasons: list[str]


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(document) for document in documents]
        self.document_lengths = [len(document) for document in self.documents]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths) if self.document_lengths else 0.0
        )
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))

    def score(self, query: str, document_index: int) -> float:
        if document_index < 0 or document_index >= len(self.documents):
            return 0.0
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        score = 0.0
        document_length = self.document_lengths[document_index]
        frequencies = self.term_frequencies[document_index]
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency <= 0:
                continue
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(1e-9, self.average_document_length)
            )
            score += self.idf(term) * term_frequency * (self.k1 + 1) / denominator
        return score

    def idf(self, term: str) -> float:
        document_count = max(1, len(self.documents))
        frequency = self.document_frequency.get(term, 0)
        return math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))


def retrieve_bm25_candidates(
    bm25: BM25Index,
    query: str,
    candidate_indices: set[int],
    *,
    reference_indices: list[int],
    top_k: int,
    temporal_next_k: int,
    min_bm25_score: float,
    bm25_weight: float,
    temporal_weight: float,
) -> list[BM25Candidate]:
    if not candidate_indices:
        return []
    raw_scores = {index: bm25.score(query, index) for index in candidate_indices}
    temporal_indices = sorted(candidate_indices)[: max(0, temporal_next_k)]
    bm25_indices = [
        index
        for index, score in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        if score >= min_bm25_score
    ][: max(0, top_k)]
    selected_indices = set(temporal_indices) | set(bm25_indices)
    if not selected_indices:
        return []
    max_bm25 = max((raw_scores[index] for index in selected_indices), default=0.0)
    candidates: list[BM25Candidate] = []
    for index in selected_indices:
        bm25_score = raw_scores[index]
        normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
        distance = min(abs(index - reference) for reference in reference_indices) if reference_indices else 0
        temporal_prior = math.exp(-distance / 20.0)
        combined_score = bm25_weight * normalized_bm25 + temporal_weight * temporal_prior
        reasons = []
        if index in bm25_indices:
            reasons.append("bm25_top_k")
        if index in temporal_indices:
            reasons.append("temporal_next_k")
        candidates.append(BM25Candidate(index, bm25_score, temporal_prior, combined_score, reasons))
    return sorted(candidates, key=lambda item: (-item.combined_score, item.episode_index))
