#!/usr/bin/env python3
"""从用户下载的官方 REALTALK 数据制作本文的 728 题测试输入；仅依赖标准库。"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil


SOURCE_REVISION = "b903e06a9770bf4e5fe9018c3e132889666d3b4a"
REVISION = "full-caption-time-r1-20260924"
SOURCE_HASHES = {
    "Chat_10_Fahim_Muhhamed.json":
        "26135a9e3482f40bea4c71de570a2d994a361848111067e60587be1dd82a0127",
    "Chat_1_Emi_Elise.json": "4e947998a751a1ba2c3710fc873b90fbcbc234fda7a5e4e75db2b762e0f25a36",
    "Chat_2_Kevin_Elise.json": "4ac6e319c206731aa9a025c93325eb1edb43686cf39473f058c87ae7deb2d62a",
    "Chat_3_Kevin_Paola.json": "10c9208016538cbe7a28aafa6cfd72587b072677964c36ec242159fdf58961f2",
    "Chat_4_Emi_Paola.json": "80aa012ed6a918f5b317bd942b4d4811a158d27a6e07b7a5c6182994107a9367",
    "Chat_5_Nicolas_Nebraas.json":
        "32dd7c45786b1a26861dc1574a1bb698a855196fd2ae3fc0c091f3c9ab2d6d06",
    "Chat_6_Vanessa_Nicolas.json":
        "9d5ee072ab5a0667ef71d5644d7cd3a4c6777a1e6e2252bf306f310d982e40ac",
    "Chat_7_Nebraas_Vanessa.json":
        "b5716bb093217980e747dec0d98996829b569febc09587a025efc6b79806e9ce",
    "Chat_8_Akib_Muhhamed.json": "814f64ad7e8a46f0398f550fe84184e2ba495a292290b9ff96ffc95448280800",
    "Chat_9_Fahim_Akib.json": "05b9b4331bf0e25a2a64dc89c404622c83a61ab5e7c935fb7d844449caf6dd39",
}
# LoCoMo 官方回答指令及本文 reasoning wrapper；保持已有评测输入原文。
RESPONSE_INSTRUCTION = (
    "Based on the above context, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context whenever possible."
    """

Before giving the short answer, provide concise reasoning based on the context.

Question: {question}

Output exactly this format:
<reasoning>
[concise reasoning based on the context]
</reasoning>
<answer>
[short answer]
</answer>"""
)
COMPARISON_FIELDS = (
    "question", "answer", "accepted_answers", "answer_aliases", "category", "evidence",
)


def render_context(source: dict) -> str:
    numbers = sorted(int(key[8:]) for key in source if re.fullmatch(r"session_\d+", key))
    names = dict.fromkeys(turn["speaker"] for n in numbers for turn in source[f"session_{n}"])
    aliases = {name: chr(65 + i) for i, name in enumerate(names)}
    legend = "; ".join(f"{value}={name}" for name, value in aliases.items())
    lines = [
        "Full conversation. Speaker labels: " + legend + ".\n"
        "Dates use DD.MM.YYYY. Each turn starts with its exact HH:MM:SS time "
        "and speaker label. "
        "A Date line applies to subsequent turns until the next Date line. "
        "[Image: ...] preserves the source image caption for that turn.\n"
    ]
    for number in numbers:
        lines.append(f"\nSession {number} starts {source[f'session_{number}_date_time']}\n")
        last_date = None
        for turn in source[f"session_{number}"]:
            text = turn["clean_text"] if "clean_text" in turn else turn["text"]
            date, time = turn["date_time"].split(", ")
            if date != last_date:
                lines.append(f"Date {date}\n")
                last_date = date
            lines.append(f"{time} {aliases[turn['speaker']]}: {text}\n")
            if "blip_caption" in turn:
                lines.append(f"[Image: {turn['blip_caption']}]\n")
    return "".join(lines)


def write_rows(path: Path, rows: list[dict]) -> str:
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0
    ) as handle:
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="官方 REALTALK 仓库固定 revision 的 data 目录")
    parser.add_argument("--output-dir", type=Path, default=Path("data/realtalk"))
    args = parser.parse_args()
    corrections = json.loads(Path(__file__).with_name("reference-fixes.json").read_text())
    fixes = {item["question_id"]: item for item in corrections}
    sources = {}
    for name, expected in SOURCE_HASHES.items():
        raw = (args.source_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{name}: source differs from REALTALK {SOURCE_REVISION}")
        sources[name] = json.loads(raw)

    rows, before, changes, decisions = [], [], [], []
    for name, source in sources.items():
        conversation_id = Path(name).stem
        context = render_context(source)
        turns = {turn["dia_id"]: turn for key in source if re.fullmatch(r"session_\d+", key)
                 for turn in source[key]}
        for index, qa in enumerate(source["qa"]):
            qid = f"realtalk_{conversation_id}_qa{index}"
            row = {
                "question_id": qid, "conversation_id": conversation_id,
                "question": qa["question"], "answer": qa["answer"],
                "accepted_answers": [qa["answer"]], "answer_aliases": [],
                "category": int(qa["category"]),
                "question_type": {1: "multi-hop", 2: "temporal", 3: "commonsense"}[
                    int(qa["category"])
                ],
                "evidence": qa.get("evidence", []), "revision": REVISION,
                "benchmark": "realtalk", "split": "test",
                "messages": [{"role": "user", "content": context + "\n\n"
                              + RESPONSE_INSTRUCTION.format(question=qa["question"])}],
                "chat_template_enable_thinking": False, "answer_protocol": "reasoning_answer",
                "context_window": 40960, "max_new_tokens": 2560,
            }
            original = {field: row[field] for field in COMPARISON_FIELDS}
            before.append({
                "question_id": qid, "source_file": "realtalk/source-original/" + name,
                "source_revision": SOURCE_REVISION, "source_question_index": index,
                "conversation_id": conversation_id, "split": "test", **original,
            })
            if qid in fixes:
                fix = fixes[qid]
                row.update(answer=fix["accepted_answers"][0],
                           accepted_answers=fix["accepted_answers"],
                           answer_aliases=fix["answer_aliases"])
                decision = {key: value for key, value in fix.items()
                            if key != "supporting_turn_ids"}
                decision.update(source_qa=qa,
                                supporting_turns=[turns[key] for key in fix["supporting_turn_ids"]])
                decisions.append(decision)
            delta = {key: {"before": original[key], "after": row[key]}
                     for key in COMPARISON_FIELDS if original[key] != row[key]}
            changes.append({"question_id": qid, "status": "revised" if delta else "unchanged",
                            "released_split": "test", "changes": delta})
            rows.append(row)

    if len(rows) != 728 or len({row["question_id"] for row in rows}) != 728:
        raise ValueError("Expected 728 unique REALTALK questions")
    if {row["question_id"] for row in decisions} != set(fixes):
        raise ValueError("Reference corrections do not match the source questions")
    output = args.output_dir
    (output / "source-original").mkdir(parents=True, exist_ok=True)
    (output / "revisions").mkdir(exist_ok=True)
    for name in SOURCE_HASHES:
        shutil.copyfile(args.source_dir / name, output / "source-original" / name)
    files = {name: write_rows(output / name, values) for name, values in (
        ("test.jsonl.gz", rows), ("before.jsonl.gz", before), ("changes.jsonl.gz", changes),
    )}
    (output / "revisions/reference-decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_repository": "https://github.com/danny911kr/REALTALK",
        "source_revision": SOURCE_REVISION, "upstream_file_sha256": SOURCE_HASHES,
        "revision": REVISION, "rows": len(rows), "reference_revisions": len(decisions),
        "categories": dict(Counter(row["category"] for row in rows)), "sha256": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} REALTALK questions in {output}")


if __name__ == "__main__":
    main()
