import argparse
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypedDict, cast

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from ranx import Qrels

from src.data import CorpusDocument, load_corpus, load_corpus_by_id
from src.io_utils import append_jsonl, load_jsonl, write_json, write_jsonl
import src.paths as paths


DEFAULT_SEED = 461
MODEL_NAME = "gemma-4-31b-it"
TARGET_OUTPUT_TOKENS = (500, 750, 1000, 1250, 1500, 1750, 2000)


ParaphraseLevel = Literal["light", "rewrite", "full_rephrase"]


class PlanRow(TypedDict):
    item_id: str
    source_doc_ids: list[str]
    target_output_tokens: int
    paraphrase_level: ParaphraseLevel


class ParaphraseConfig(TypedDict):
    weight: float
    instruction: str


PARAPHRASE_LEVEL_CONFIGS: dict[ParaphraseLevel, ParaphraseConfig] = {
    "light": {
        "weight": 0.6,
        "instruction": "Stay fairly close to the source wording and structure where natural.",
    },
    "rewrite": {
        "weight": 0.3,
        "instruction": "Modify sentence structure, reword phrases, and incorporate elements of general knowledge to ensure coherence. The less token overlap, the better.",
    },
    "full_rephrase": {
        "weight": 0.1,
        "instruction": "Completely rephrase in your own words. Feel free to incorporate elements from general knowledge to ensure coherence, flow, and better understanding.",
    },
}
BASE_PROMPT = """You are given several source documents.

Write a survey article that synthesizes material from all of them into one coherent review. The survey should organize, compare, and contextualize the source documents rather than present itself as a single original study.

Use the source documents as the content basis for the survey, drawing on their ideas, methods, findings, terminology, and examples where relevant. Group related themes, highlight similarities and differences, and explain how the documents relate to one another. Do not invent new experiments, results, or claims of original authorship.

Output the survey in Markdown using exactly these section headings and in this order:

## Abstract
## Introduction
## Methodology
## Experiment
## Discussion
## Conclusion

Start directly with ## Abstract.
Always put exactly one blank line after every heading.
Prefer plain prose and use math only sparingly and preferably inline
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def job_dir(job_name: str) -> Path:
    return paths.SYNTHETIC_DATA_DIR / job_name


def reserve_job_dir(job_name: str) -> Path:
    paths.SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = job_dir(job_name)
    try:
        path.mkdir(parents=True)
    except FileExistsError as error:
        raise ValueError(f"Job already exists: {job_name}") from error
    return path


def sample_source_count(rng: random.Random) -> int:
    return max(2, min(4, round(rng.gauss(2.0, 1.5))))


def load_completed_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(cast(dict[str, object], row)["qid"]) for row in load_jsonl(path)}


def append_failure(path: Path, item_id: str, error: str) -> None:
    append_jsonl(path, {"item_id": item_id, "error": error})


class GoogleAiStudioClient:
    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    def generate_text(self, prompt: str, target_output_tokens: int, seed: int) -> str:
        response = self._client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"max_output_tokens": target_output_tokens, "seed": seed},
        )
        return response.text or ""


def create_plan_rows(corpus_rows: Sequence[CorpusDocument], job_name: str, count: int) -> list[PlanRow]:
    if len(corpus_rows) < 2:
        raise ValueError("The corpus must contain at least two rows to create a plan.")

    rng = random.Random(DEFAULT_SEED)
    doc_ids = [row.doc_id for row in corpus_rows]
    plan_rows: list[PlanRow] = []

    for index in range(count):
        source_count = min(sample_source_count(rng), len(doc_ids))
        paraphrase_level = cast(
            ParaphraseLevel,
            rng.choices(
                tuple(PARAPHRASE_LEVEL_CONFIGS),
                weights=tuple(config["weight"] for config in PARAPHRASE_LEVEL_CONFIGS.values()),
                k=1,
            )[0],
        )
        plan_rows.append(
            {
                "item_id": f"{job_name}-{index + 1:04d}",
                "source_doc_ids": rng.sample(doc_ids, source_count),
                "target_output_tokens": rng.choice(TARGET_OUTPUT_TOKENS),
                "paraphrase_level": paraphrase_level,
            }
        )

    return plan_rows


def build_prompt(
    source_rows: Sequence[CorpusDocument],
    target_output_tokens: int,
    paraphrase_level: ParaphraseLevel,
) -> str:
    source_blocks = [
        f"### Source {index}: {row.doc_id}\n{row.default_text.strip()}"
        for index, row in enumerate(source_rows, start=1)
    ]

    return "\n\n".join(
        [
            BASE_PROMPT.strip(),
            f"Write the survey so it is roughly {target_output_tokens} tokens long.",
            PARAPHRASE_LEVEL_CONFIGS[paraphrase_level]["instruction"],
            "Source documents:",
            "\n\n".join(source_blocks),
        ]
    )


def write_job_metadata(job_name: str, job_path: Path, count: int) -> None:
    write_json(
        job_path,
        {
            "job_name": job_name,
            "count": count,
            "seed": DEFAULT_SEED,
            "model": MODEL_NAME,
            "created_at": now_iso(),
        },
        indent=2,
    )


def run_plan(job_name: str, count: int) -> Path:
    path = reserve_job_dir(job_name)
    job_path = path / "job.json"
    plan_path = path / "plan.jsonl"
    qrels_path = path / "qrels.txt"
    plan_rows = create_plan_rows(load_corpus(paths.CORPUS_PATH), job_name, count)
    write_job_metadata(job_name, job_path, count)
    write_jsonl(plan_path, plan_rows)
    write_plan_qrels(qrels_path, plan_rows)
    return plan_path


def write_plan_qrels(path: Path, plan_rows: Sequence[PlanRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not plan_rows:
        path.write_text("", encoding="utf-8")
        return

    qrels_by_qid: dict[str, dict[str, int]] = {}
    for plan_row in plan_rows:
        qrels_by_qid[plan_row["item_id"]] = {doc_id: 1 for doc_id in plan_row["source_doc_ids"]}

    Qrels(qrels_by_qid).save(str(path))


def generate_query_row(
    plan_row: PlanRow,
    source_rows: Sequence[CorpusDocument],
    client: GoogleAiStudioClient,
) -> dict[str, str | int]:
    generated_text = client.generate_text(
        build_prompt(source_rows, plan_row["target_output_tokens"], plan_row["paraphrase_level"]),
        plan_row["target_output_tokens"],
        DEFAULT_SEED,
    ).strip()
    if not generated_text:
        raise RuntimeError("Model response did not include text.")

    return {
        "qid": plan_row["item_id"],
        "query": generated_text,
        "word_count": len(generated_text.split()),
        "paraphrase_level": plan_row["paraphrase_level"],
    }


def generate_pending_items(
    plan_rows: Sequence[PlanRow],
    corpus_by_id: Mapping[str, CorpusDocument],
    output_path: Path,
    failures_path: Path,
    client: GoogleAiStudioClient,
    max_items_this_run: int | None,
) -> int:
    # The output file is the source of truth for resumability. If a row is
    # already there, we treat it as done and move on.
    completed_qids = load_completed_qids(output_path)
    skipped_count = sum(1 for plan_row in plan_rows if plan_row["item_id"] in completed_qids)

    generated_count = 0

    if skipped_count:
        print(f"[skip] {skipped_count} completed items already exist in {output_path.name}")

    for plan_row in plan_rows:
        item_id = plan_row["item_id"]
        if item_id in completed_qids:
            continue
        if max_items_this_run is not None and generated_count >= max_items_this_run:
            break

        missing_doc_ids = [doc_id for doc_id in plan_row["source_doc_ids"] if doc_id not in corpus_by_id]
        if missing_doc_ids:
            error_message = f"Missing source docs: {', '.join(missing_doc_ids)}"
            append_failure(failures_path, item_id, error_message)
            print(f"[fail] {item_id} missing source docs: {', '.join(missing_doc_ids)}")
            continue

        print(
            f"[start] {item_id} docs={len(plan_row['source_doc_ids'])} "
            f"target_tokens={plan_row['target_output_tokens']} "
            f"paraphrase={plan_row['paraphrase_level']}"
        )
        source_rows = [corpus_by_id[doc_id] for doc_id in plan_row["source_doc_ids"]]
        started_at = time.perf_counter()
        try:
            output_row = generate_query_row(plan_row, source_rows, client)
        except (genai_errors.APIError, RuntimeError) as error:
            append_failure(failures_path, item_id, str(error))
            print(f"[fail] {item_id} {error}")
            continue

        append_jsonl(output_path, output_row)
        completed_qids.add(item_id)
        generated_count += 1
        elapsed_seconds = time.perf_counter() - started_at
        print(f"[done] {item_id} words={output_row['word_count']} elapsed={elapsed_seconds:.2f}s")

    return generated_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and run synthetic data jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Create a synthetic generation plan.")
    plan_parser.add_argument("job_name")
    plan_parser.add_argument("--count", type=int, required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data.")
    generate_parser.add_argument("--job-name", required=True)
    generate_parser.add_argument("--max-items-this-run", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.command == "plan":
        try:
            plan_path = run_plan(args.job_name, args.count)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"Job name: {args.job_name}")
        print(f"Plan path: {plan_path}")
        return 0

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is required for generate runs.")

    path = job_dir(args.job_name)
    plan_path = path / "plan.jsonl"
    queries_path = path / "queries.jsonl"
    qrels_path = path / "qrels.txt"
    failures_path = path / "failures.jsonl"
    plan_rows = cast(list[PlanRow], load_jsonl(plan_path))
    completed_qids = load_completed_qids(queries_path)
    corpus_by_id = load_corpus_by_id(paths.CORPUS_PATH)
    client = GoogleAiStudioClient(api_key)

    remaining_count = max(0, len(plan_rows) - len(completed_qids))

    print(f"Generating job {args.job_name}")
    print(f"Plan items: {len(plan_rows)} | already complete: {len(completed_qids)} | remaining: {remaining_count}")
    print(f"Queries path: {queries_path}")
    print(f"Qrels path: {qrels_path}")
    print(f"Failures path: {failures_path}")

    generated_count = generate_pending_items(
        plan_rows,
        corpus_by_id,
        queries_path,
        failures_path,
        client,
        args.max_items_this_run,
    )
    print(f"Generate run finished. Wrote {generated_count} new rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
