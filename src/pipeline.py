# Wire the retrieval stages together and write one experiment directory

import shutil
from collections.abc import Callable
from typing import cast
from dataclasses import dataclass, field
from pathlib import Path

from ranx import Run, evaluate, Qrels

from src.bm25 import Bm25Config
from src.data import QueryRecord, load_qrels, load_queries
from src.io_utils import write_json, write_jsonl
from src.paths import CORPUS_PATH, INDICES_DIR, RETRIEVAL_RUNS_DIR
from src.io_utils import iter_with_progress
from src.retrieval_backend import RetrievalBackendConfig, resolve_index_dir
from src.span_fusion import ScoreNormalization, SpanFusionMethod, fuse_rankings


DEFAULT_METRICS = (
    "ndcg@10",
    "recall@10",
    "recall@100",
    "map",
    "mrr",
    "precision@10",
    "ndcg@100",
    "recall@1000",
)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    query_path: Path
    corpus_path: Path = CORPUS_PATH
    qrels_path: Path | None = None
    index_root: Path = INDICES_DIR
    output_root: Path = RETRIEVAL_RUNS_DIR
    run_depth: int = 100
    metrics: tuple[str, ...] = DEFAULT_METRICS
    backend: RetrievalBackendConfig = field(default_factory=Bm25Config)
    span_selector: Callable[[str], list[str]] | None = None
    rrf_k: int = 60
    span_fusion_method: SpanFusionMethod = "rrf"
    span_score_normalization: ScoreNormalization = "none"
    span_fusion_rank_cutoff: int | None = None

    @property
    def experiment_dir(self) -> Path:
        return self.output_root / self.experiment_name


def run_experiment(config: ExperimentConfig) -> Path:
    index_dir = resolve_index_dir(config.corpus_path, config.index_root, config.backend)
    experiment_dir = config.experiment_dir
    run_dir = experiment_dir / "run"
    eval_dir = experiment_dir / "eval"
    run_path = experiment_dir / "run" / "run.trec"

    _prepare_experiment(config, index_dir, run_dir, eval_dir)
    try:
        config.backend.ensure_index(config.corpus_path, index_dir)
        suspicious_documents = load_queries(config.query_path)
        selected_spans_by_qid = _select_query_spans(
            suspicious_documents,
            config.span_selector,
        )
        span_queries = [span_query for span_list in selected_spans_by_qid.values() for span_query in span_list]
        per_span_rankings = config.backend.retrieve(span_queries, index_dir, config.run_depth)
        rankings: dict[str, dict[str, float]] = {}
        for qid, span_list in iter_with_progress(
            selected_spans_by_qid.items(),
            desc="Fusing query span rankings",
            total=len(selected_spans_by_qid),
        ):
            ranking = _build_fused_ranking(
                span_list,
                per_span_rankings,
                config.run_depth,
                config.rrf_k,
                config.span_fusion_method,
                config.span_score_normalization,
                config.span_fusion_rank_cutoff,
            )
            rankings[qid] = ranking

        run = Run(rankings, name=config.experiment_name)
        run.save(str(run_path))
        _write_evaluation_outputs(config, run, eval_dir)
        return experiment_dir
    finally:
        config.backend.release_resources()


def _prepare_experiment(
    config: ExperimentConfig,
    index_dir: Path,
    run_dir: Path,
    eval_dir: Path,
) -> None:
    experiment_dir = config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(run_dir, eval_dir)
    for directory in (run_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_json(
        experiment_dir / "experiment.json",
        {
            "experiment_name": config.experiment_name,
            "corpus_path": str(config.corpus_path),
            "query_path": str(config.query_path),
            "qrels_path": str(config.qrels_path) if config.qrels_path is not None else None,
            "index_root": str(config.index_root),
            "index_dir": str(index_dir),
            "output_root": str(config.output_root),
            "run_depth": config.run_depth,
            "metrics": list(config.metrics),
            "backend": config.backend.to_metadata(),
            "span_selector": {"enabled": config.span_selector is not None},
            "fusion": {
                "method": config.span_fusion_method,
                "k": config.rrf_k if config.span_fusion_method == "rrf" else None,
                "normalization": config.span_score_normalization
                if config.span_fusion_method in {"comb_sum", "comb_mnz"}
                else None,
                "rank_cutoff": config.span_fusion_rank_cutoff,
            },
        },
        indent=2,
    )


def _write_evaluation_outputs(
    config: ExperimentConfig,
    run: Run,
    eval_dir: Path,
) -> None:
    if config.qrels_path is None or not config.qrels_path.exists():
        return

    qrels = load_qrels(config.qrels_path)
    summary, per_query = evaluate_run(run, qrels, config.metrics)
    write_json(eval_dir / "summary.json", summary, indent=2)
    write_jsonl(eval_dir / "per_query_metrics.jsonl", per_query)


def _clear_previous_outputs(run_dir: Path, eval_dir: Path) -> None:
    for directory in (run_dir, eval_dir):
        if directory.exists():
            shutil.rmtree(directory)


def _select_query_spans(
    suspicious_documents: list[QueryRecord],
    span_selector: Callable[[str], list[str]] | None,
) -> dict[str, list[QueryRecord]]:
    selected_spans_by_qid: dict[str, list[QueryRecord]] = {}

    for suspicious_document in iter_with_progress(
        suspicious_documents,
        desc="Selecting query spans",
        total=len(suspicious_documents),
    ):
        span_texts = [suspicious_document.text] if span_selector is None else span_selector(suspicious_document.text)

        selected_spans_by_qid[suspicious_document.qid] = [
            QueryRecord(
                qid=f"{suspicious_document.qid}__{index}",
                text=span_text,
                metadata=dict(suspicious_document.metadata),
            )
            for index, span_text in enumerate(span_texts)
        ]
    return selected_spans_by_qid


def _build_fused_ranking(
    span_queries: list[QueryRecord],
    per_span_rankings: dict[str, dict[str, float]],
    run_depth: int,
    rrf_k: int,
    span_fusion_method: SpanFusionMethod,
    span_score_normalization: ScoreNormalization,
    span_fusion_rank_cutoff: int | None,
) -> dict[str, float]:
    if len(span_queries) == 1:
        return _trim_results(per_span_rankings[span_queries[0].qid], run_depth)

    span_rankings = [per_span_rankings[query.qid] for query in span_queries]
    return fuse_rankings(
        span_rankings,
        method=span_fusion_method,
        normalization=span_score_normalization,
        rrf_k=rrf_k,
        run_depth=run_depth,
        rank_cutoff=span_fusion_rank_cutoff,
    )


def _trim_results(results: dict[str, float], run_depth: int) -> dict[str, float]:
    ranked_results = sorted(results.items(), key=lambda item: (-item[1], item[0]))
    return {doc_id: score for doc_id, score in ranked_results[:run_depth]}


def evaluate_run(
    run: Run,
    qrels: Qrels,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> tuple[dict[str, float | int], list[dict[str, float | str]]]:
    # align run and qrels
    qids = list(qrels.to_dict())
    comparable_run = run.make_comparable(qrels)
    # we always evaluates multiple metrics together, so we only
    # support ranx's dict-shaped multi-metric result here
    aggregate_scores = cast(
        dict[str, float],
        evaluate(qrels, comparable_run, list(metrics), make_comparable=False),
    )

    # prepare summary
    summary: dict[str, float | int] = {"query_count": len(qids)}
    for metric in metrics:
        summary[metric] = float(aggregate_scores[metric])

    # prepare per-query metrics
    scores_by_metric: dict[str, dict[str, float]] = {}
    for metric, scores_by_qid in dict(comparable_run.scores).items():
        metric_scores: dict[str, float] = {}
        for qid, score in dict(scores_by_qid).items():
            metric_scores[qid] = float(score)
        scores_by_metric[metric] = metric_scores

    # write per-query metrics
    per_query: list[dict[str, float | str]] = []
    for qid in qids:
        row: dict[str, float | str] = {"qid": qid}
        for metric in metrics:
            row[metric] = scores_by_metric[metric].get(qid, 0.0)
        per_query.append(row)

    return summary, per_query
