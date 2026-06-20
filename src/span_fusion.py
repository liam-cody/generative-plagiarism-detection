import math
from typing import Literal


SpanFusionMethod = Literal["rrf", "comb_sum", "comb_mnz"]
ScoreNormalization = Literal["minmax", "zscore", "none"]


def fuse_rankings(
    rankings: list[dict[str, float]],
    *,
    method: SpanFusionMethod,
    normalization: ScoreNormalization,
    rrf_k: int,
    run_depth: int,
    rank_cutoff: int | None = None,
) -> dict[str, float]:
    if run_depth <= 0:
        return {}
    if rank_cutoff is not None and rank_cutoff <= 0:
        return {}
    if method == "rrf":
        return _rrf(rankings, rrf_k, run_depth)
    if method in {"comb_sum", "comb_mnz"}:
        return _comb(
            rankings,
            method=method,
            normalization=normalization,
            run_depth=run_depth,
            rank_cutoff=rank_cutoff,
        )
    raise ValueError(f"Unsupported span fusion method: {method}")


def _rrf(rankings: list[dict[str, float]], rrf_k: int, run_depth: int) -> dict[str, float]:
    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        for zero_based_rank, doc_id in enumerate(_ranked_doc_ids(ranking)):
            rank = zero_based_rank + 1
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return _trim_results(fused_scores, run_depth)


def _comb(
    rankings: list[dict[str, float]],
    *,
    method: Literal["comb_sum", "comb_mnz"],
    normalization: ScoreNormalization,
    run_depth: int,
    rank_cutoff: int | None,
) -> dict[str, float]:
    summed_scores: dict[str, float] = {}
    retrieved_counts: dict[str, int] = {}

    for ranking in rankings:
        normalized_ranking = _normalize_scores(ranking, normalization)
        counted_ranking = _filter_to_top_ranked_docs(normalized_ranking, ranking, rank_cutoff)
        for doc_id, score in counted_ranking.items():
            summed_scores[doc_id] = summed_scores.get(doc_id, 0.0) + score
            retrieved_counts[doc_id] = retrieved_counts.get(doc_id, 0) + 1

    if method == "comb_sum":
        return _trim_results(summed_scores, run_depth)

    mnz_scores = {doc_id: score * retrieved_counts[doc_id] for doc_id, score in summed_scores.items()}
    return _trim_results(mnz_scores, run_depth)


def _normalize_scores(ranking: dict[str, float], normalization: ScoreNormalization) -> dict[str, float]:
    if not ranking or normalization == "none":
        return dict(ranking)

    scores = list(ranking.values())
    if normalization == "minmax":
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(max_score, min_score):
            return {doc_id: 1.0 for doc_id in ranking}
        denominator = max_score - min_score
        return {doc_id: (score - min_score) / denominator for doc_id, score in ranking.items()}

    if normalization == "zscore":
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        standard_deviation = math.sqrt(variance)
        if math.isclose(standard_deviation, 0.0):
            return {doc_id: 0.0 for doc_id in ranking}
        return {doc_id: (score - mean_score) / standard_deviation for doc_id, score in ranking.items()}

    raise ValueError(f"Unsupported score normalization: {normalization}")


def _ranked_doc_ids(ranking: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _score in sorted(ranking.items(), key=lambda item: (-item[1], item[0]))]


def _filter_to_top_ranked_docs(
    normalized_ranking: dict[str, float],
    raw_ranking: dict[str, float],
    rank_cutoff: int | None,
) -> dict[str, float]:
    if rank_cutoff is None:
        return normalized_ranking
    top_doc_ids = set(_ranked_doc_ids(raw_ranking)[:rank_cutoff])
    return {doc_id: score for doc_id, score in normalized_ranking.items() if doc_id in top_doc_ids}


def _trim_results(results: dict[str, float], run_depth: int) -> dict[str, float]:
    ranked_results = sorted(results.items(), key=lambda item: (-item[1], item[0]))
    return {doc_id: score for doc_id, score in ranked_results[:run_depth]}
