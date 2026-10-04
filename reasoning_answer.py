"""``<reasoning>`` / ``<answer>`` 回答协议的确定性抽取与严格 schema 校验。"""

from __future__ import annotations

import re

_SHORT_ANSWER_PREFIX = re.compile(r"^Short answer:\s*", flags=re.IGNORECASE)
_STRUCTURE_TAG = re.compile(
    r"<\s*/?\s*(?:reasoning|answer|think|thinking)(?=[\s>/]|$)", flags=re.IGNORECASE
)
_COMPLETE_BOUNDARY_TAG = re.compile(r"<\s*(/?)\s*(reasoning|answer)\s*>", re.IGNORECASE)


def _tag_span(raw_response: str, tag: str) -> tuple[int, int, int, int] | None:
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    if raw_response.count(opening) != 1 or raw_response.count(closing) != 1:
        return None
    opening_start = raw_response.index(opening)
    content_start = opening_start + len(opening)
    closing_start = raw_response.index(closing)
    if closing_start < content_start:
        return None
    return opening_start, content_start, closing_start, closing_start + len(closing)


def extract_reasoning(raw_response: str) -> str:
    """只抽取唯一、闭合且顺序正确的 reasoning；否则返回空字符串。"""

    span = _tag_span(raw_response, "reasoning")
    if span is None:
        return ""
    return raw_response[span[1] : span[2]].strip()


def validate_schema(raw_response: str) -> dict[str, str | bool]:
    """严格校验两个非空标签唯一、闭合、顺序正确且标签外没有文本。"""

    spans = {}
    for tag in ("reasoning", "answer"):
        opening = f"<{tag}>"
        closing = f"</{tag}>"
        opening_count = raw_response.count(opening)
        closing_count = raw_response.count(closing)
        if opening_count == 0:
            return {"format_valid": False, "format_status": f"missing_{tag}_open_tag"}
        if closing_count == 0:
            return {"format_valid": False, "format_status": f"missing_{tag}_close_tag"}
        if opening_count != 1 or closing_count != 1:
            return {"format_valid": False, "format_status": f"multiple_{tag}_tags"}

        span = _tag_span(raw_response, tag)
        if span is None:
            return {"format_valid": False, "format_status": f"malformed_{tag}_tags"}
        if not raw_response[span[1] : span[2]].strip():
            return {"format_valid": False, "format_status": f"empty_{tag}"}
        spans[tag] = span

    reasoning = spans["reasoning"]
    answer = spans["answer"]
    if reasoning[3] > answer[0]:
        return {"format_valid": False, "format_status": "wrong_section_order"}

    outside_segments = (
        raw_response[: reasoning[0]],
        raw_response[reasoning[3] : answer[0]],
        raw_response[answer[3] :],
    )
    if any(segment.strip() for segment in outside_segments):
        return {"format_valid": False, "format_status": "extra_text_outside_sections"}
    return {"format_valid": True, "format_status": "ok"}


def extract_answer(
    raw_response: str, *, normalize_boundaries: bool = False,
) -> dict[str, str | bool]:
    """按 section 边界抽取；显式 opt-in 只规范完整边界标签，strict仍检查原文。"""

    original_response = raw_response
    if normalize_boundaries:
        # 仅修复完整、无属性的标签；不猜未闭合标签或没有答案边界的 reasoning。
        # 先统一再计数，因此不同大小写/空白的多答案仍会被拒绝。
        raw_response = _COMPLETE_BOUNDARY_TAG.sub(
            lambda match: f"<{match.group(1)}{match.group(2).lower()}>", raw_response,
        )

    opening = "<answer>"
    closing = "</answer>"
    opening_count = raw_response.count(opening)
    closing_count = raw_response.count(closing)

    answer = ""
    success_status = ""
    failure_status = ""
    if opening_count == 1 and closing_count == 1:
        opening_start = raw_response.index(opening)
        content_start = opening_start + len(opening)
        closing_start = raw_response.index(closing)
        if closing_start < content_start:
            failure_status = "wrong_answer_tag_order"
        else:
            answer = raw_response[content_start:closing_start].strip()
            success_status = "ok_closed_answer"
    elif opening_count == 1 and closing_count == 0:
        content_start = raw_response.index(opening) + len(opening)
        answer = raw_response[content_start:].strip()
        success_status = "ok_open_answer_to_eos"
    elif opening_count == 0 and closing_count == 0:
        reasoning_closing = "</reasoning>"
        if reasoning_closing not in raw_response:
            if raw_response.strip() and not _STRUCTURE_TAG.search(raw_response):
                answer = raw_response.strip()
                success_status = "ok_raw_fallback"
            else:
                failure_status = "missing_answer_boundary"
        else:
            answer = raw_response.rsplit(reasoning_closing, 1)[1].strip()
            success_status = "ok_after_reasoning_to_eos"
    elif opening_count == 0:
        failure_status = "orphan_answer_close_tag"
    elif opening_count > 1 or closing_count > 1:
        failure_status = "multiple_answer_tags"
    else:
        failure_status = "malformed_answer_tags"

    if answer and success_status != "ok_raw_fallback":
        answer = _SHORT_ANSWER_PREFIX.sub("", answer, count=1).strip()

    if failure_status:
        answer_result = {
            "answer": "",
            "answer_extract_valid": False,
            "status": failure_status,
        }
    elif not answer:
        answer_result = {
            "answer": "",
            "answer_extract_valid": False,
            "status": "empty_answer",
        }
    else:
        answer_result = {
            "answer": answer,
            "answer_extract_valid": True,
            "status": success_status,
        }

    return {
        **answer_result,
        "reasoning": extract_reasoning(raw_response),
        **validate_schema(original_response),
    }
