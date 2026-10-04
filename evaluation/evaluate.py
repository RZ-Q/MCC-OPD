#!/usr/bin/env python3
"""Score saved answers on the four paper benchmarks, with optional API judging."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reasoning_answer import extract_answer
from evaluation.metrics import score_answer
from evaluation.prompts import judge_prompt, parse_judge
from evaluation import rubrics


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def request_judge(prompt: str, *, url: str, model: str, api_key: str) -> str:
    response = requests.post(
        url, headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": 2048, "thinking": {"type": "disabled"}},
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Judge response has no text content")
    return content


def judge_one(job: dict, args, api_key: str) -> dict:
    """每票独立请求；只有前两票分歧时才请求第三票，按完整输入恢复。"""
    path = args.output_dir / "votes" / f"{job['judge_input_sha256']}.json"
    record = json.loads(path.read_text()) if path.exists() else {
        "prompt": job["judge_prompt"], "prompt_version": job["judge_prompt_version"],
        "model": args.judge_model, "endpoint": args.judge_url, "votes": [], "errors": [],
    }
    for index in range(3):
        if index == 2 and record["votes"][0]["correct"] == record["votes"][1]["correct"]:
            break
        if len(record["votes"]) > index:
            continue
        for attempt in range(args.max_attempts):
            raw = None
            try:
                raw = request_judge(
                    job["judge_prompt"], url=args.judge_url,
                    model=args.judge_model, api_key=api_key,
                )
                correct = parse_judge(args.benchmark, raw)
                record["votes"].append({"index": index + 1, "correct": correct,
                                         "raw_response": raw})
                save_json(path, record)
                break
            except (requests.RequestException, ValueError, KeyError, IndexError) as error:
                record["errors"].append({"vote": index + 1, "type": type(error).__name__,
                                          "raw_response": raw})
                save_json(path, record)
                if attempt + 1 < args.max_attempts:
                    time.sleep(2 ** (attempt + 1))
        else:
            return {"status": "missing", "correct": None,
                    "vote_count": len(record["votes"])}
    votes = record["votes"]
    return {"status": "judged", "correct": sum(v["correct"] for v in votes) > len(votes) / 2,
            "vote_count": len(votes),
            "decision_source": "third_vote_majority" if len(votes) == 3 else "two_vote_agreement"}


def aggregate(rows: list[dict]) -> dict:
    result = {"questions": len(rows),
              **{key: 100 * statistics.mean(r[key] for r in rows)
                 for key in ("f1", "em", "bleu1")},
              "empty_answers": sum(not r["answer"].strip() for r in rows)}
    if "judge" in rows[0]:
        decisions = [r["judge"]["correct"] for r in rows]
        result["missing_judgments"] = sum(v is None for v in decisions)
        result["accuracy"] = (100 * statistics.mean(decisions)
                              if all(v is not None for v in decisions) else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True,
                        choices=("locomo", "realtalk", "musique", "longmemeval"))
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--judge", action="store_true", help="request two votes; a third on disagreement")
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument("--judge-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1:
        parser.error("workers and max-attempts must be positive")
    references = read_jsonl(args.references)
    predictions = read_jsonl(args.predictions)
    by_id = {str(r["question_id"]): r for r in predictions}
    reference_ids = [str(r["question_id"]) for r in references]
    if (not references or len(by_id) != len(predictions)
            or len(set(reference_ids)) != len(references) or set(by_id) != set(reference_ids)):
        raise ValueError("predictions must contain every reference ID exactly once")
    if any(r["benchmark"] != args.benchmark for r in references):
        raise ValueError("reference benchmark does not match --benchmark")
    if any(path.resolve() in {(args.output_dir / "scored.jsonl").resolve(),
                              (args.output_dir / "summary.json").resolve()}
           for path in (args.references, args.predictions)):
        raise ValueError("output files must differ from reference and prediction inputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], {}
    for reference in references:
        qid = str(reference["question_id"])
        prediction = by_id[qid]
        extraction = extract_answer(
            prediction["raw_response"],
            normalize_boundaries=args.benchmark in {"locomo", "longmemeval"},
        )
        answer = str(extraction["answer"])
        row = {"question_id": qid,
               "question_type": reference.get("question_type", str(reference.get("hop_count"))),
               "abstention": "_abs" in qid,
               "answer": answer, "answer_extraction_status": extraction["status"],
               "format_valid": extraction["format_valid"],
               **score_answer(args.benchmark, reference, answer)}
        if args.judge:
            prompt, version = judge_prompt(args.benchmark, reference, answer)
            identity = {"benchmark": args.benchmark, "model": args.judge_model,
                        "endpoint": args.judge_url, "prompt": prompt, "prompt_version": version,
                        "parser": inspect.getsource(rubrics.parse_official_label
                                                     if args.benchmark == "longmemeval"
                                                     else rubrics.parse_label),
                        "temperature": 0, "max_tokens": 2048, "thinking": False}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            row.update(judge_input_sha256=digest, judge_prompt_version=version,
                       judge_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            if answer.strip():
                jobs[digest] = {**row, "judge_prompt": prompt}
        rows.append(row)
    if args.judge:
        api_key = os.environ["JUDGE_API_KEY"] if jobs else ""
        (args.output_dir / "votes").mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(lambda job: judge_one(job, args, api_key), jobs.values())
            decisions = dict(zip(jobs, results))
        for row in rows:
            row["judge"] = decisions[row["judge_input_sha256"]] if row["answer"].strip() else {
                "status": "judged", "correct": False, "vote_count": 0,
                "decision_source": "empty_answer_rule",
            }
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)
    summary = {"benchmark": args.benchmark, "metric_scale": "percentage 0..100",
               "overall": aggregate(rows),
               "by_question_type": {name: aggregate(items) for name, items in sorted(groups.items())},
               "references_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest(),
               "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    if args.judge:
        summary.update(judge_model=args.judge_model, judge_endpoint=args.judge_url)
        if args.benchmark == "longmemeval":
            values = [group["accuracy"] for group in summary["by_question_type"].values()]
            summary["task_average_accuracy"] = (statistics.mean(values)
                                                if len(values) == 6 and None not in values else None)
            abstentions = [r for r in rows if r["abstention"]]
            summary["abstention"] = aggregate(abstentions) if abstentions else None
    with (args.output_dir / "scored.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["overall"]))
    if summary["overall"].get("missing_judgments", 0):
        raise SystemExit("Some votes failed; rerun the same command to resume saved votes.")


if __name__ == "__main__":
    main()
