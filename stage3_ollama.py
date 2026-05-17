#!/usr/bin/env python3
"""Stage 3: Add answers + short solution summaries to extracted question JSON."""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import sys


ANSWER_SYSTEM_PROMPT = """
You are an accurate mathematics solver.

Return ONLY one JSON object with this exact schema:
{
  "answer": "final short answer",
  "summary": "brief step-by-step reasoning"
}

Rules:
1. Always return valid JSON and nothing else.
2. For MCQ, "answer" must identify the correct option text (or option label + text if label is visible).
3. For non-MCQ, "answer" should be the final result/value.
4. "summary" must explain key steps succinctly.
5. If the question is unclear, set "answer" to "Unknown" and explain why in "summary".
6. Treat mathematical content as LaTeX-aware text. Correctly interpret expressions like \\sin^2\\theta, fractions, roots, limits, matrices, etc.
7. Never return placeholders such as "Your answer here", "A brief summary of the content", or schema templates.
""".strip()


@dataclass
class OllamaAnswerConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b"
    timeout_sec: int = 600
    num_ctx: int = 32768
    num_predict: int = 1024
    temperature: float = 0.0


def _extract_first_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from model.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("Model output does not contain a valid JSON object.")

    # Find the first balanced top-level JSON object.
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end == -1:
        raise ValueError("Model output does not contain a complete JSON object.")
    return json.loads(text[start : end + 1])


def _request_ollama_answer(config: OllamaAnswerConfig, question_payload: Dict[str, Any]) -> Dict[str, str]:
    url = f"{config.base_url.rstrip('/')}/api/generate"
    question = question_payload.get("question", {}) or {}
    qtext = str(question.get("text", "") or "").strip()
    options = question.get("options", []) or []
    options_block = "\n".join(f"- {opt}" for opt in options) if options else "(none)"
    is_mcq = bool(options)

    if is_mcq:
        prompt = (
            "Solve this MCQ and return only the required JSON schema.\n"
            "The question text may contain LaTeX notation; interpret it mathematically.\n\n"
            f"Question text:\n{qtext}\n\n"
            f"Options:\n{options_block}\n\n"
            'Return exactly: {"answer":"...","summary":"..."}'
        )
    else:
        prompt = (
            "Solve this question and return only the required JSON schema.\n"
            "The question text may contain LaTeX notation; interpret it mathematically.\n\n"
            f"Question text:\n{qtext}\n\n"
            'Return exactly: {"answer":"...","summary":"..."}'
        )

    def _looks_like_placeholder(answer: str, summary: str) -> bool:
        joined = f"{answer} {summary}".strip().lower()
        bad_patterns = [
            "your answer here",
            "brief summary of the content",
            "lorem ipsum",
            "placeholder",
            "example summary",
            "example answer",
        ]
        return any(pattern in joined for pattern in bad_patterns)

    def _is_weak_answer(answer: str, summary: str) -> bool:
        a = (answer or "").strip().lower()
        s = (summary or "").strip().lower()
        weak_answer = a in {"", "unknown", "n/a", "none", "not sure"}
        weak_summary = s in {"", "no summary generated.", "unknown", "n/a"}
        return weak_answer and weak_summary

    def _normalize_mcq_answer_to_index(answer_text: str, option_values: List[str]) -> str:
        txt = (answer_text or "").strip()
        if not txt:
            return ""

        m_digit = re.search(r"\b([1-4])\b", txt)
        if m_digit:
            return m_digit.group(1)

        m_label = re.search(r"\b([A-Da-d])\b", txt)
        if m_label:
            return str(ord(m_label.group(1).upper()) - ord("A") + 1)

        def _canon(val: str) -> str:
            lowered = val.lower().strip()
            lowered = re.sub(r"^\([a-d]\)\s*", "", lowered)
            lowered = re.sub(r"\s+", " ", lowered)
            return lowered

        answer_canon = _canon(txt)
        for idx, opt in enumerate(option_values, start=1):
            opt_canon = _canon(str(opt))
            if answer_canon == opt_canon or answer_canon in opt_canon or opt_canon in answer_canon:
                return str(idx)
        return ""

    def _parse_answer_summary_text(raw_text: str) -> Optional[Dict[str, str]]:
        text = (raw_text or "").strip()
        if not text:
            return None
        answer_match = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", text)
        summary_match = re.search(r"(?im)^\s*summary\s*:\s*(.+)$", text)
        if answer_match and summary_match:
            return {
                "answer": answer_match.group(1).strip(),
                "summary": summary_match.group(1).strip(),
            }
        # Fallback: first non-empty line as answer, rest as summary.
        lines = [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        if len(lines) == 1:
            return {"answer": lines[0], "summary": "Model returned a single-line answer."}
        return {"answer": lines[0], "summary": " ".join(lines[1:])}

    debug = bool(question_payload.get("_debug", False))
    question_debug_id = str(question_payload.get("_debug_question_id", "q?"))
    req_counter = 0

    def _debug_log(message: str) -> None:
        if debug:
            print(message, file=sys.stderr, flush=True)

    def _generate_raw(system: str, user_prompt: str, json_mode: bool = True) -> str:
        nonlocal req_counter
        req_counter += 1
        payload = {
            "model": config.model,
            "system": system,
            "prompt": user_prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": config.temperature,
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"
        started_at = time.perf_counter()
        _debug_log(
            f"[stage3][{question_debug_id}][req#{req_counter}] REQUEST ({'json' if json_mode else 'text'})\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        resp = requests.post(url, json=payload, timeout=config.timeout_sec)
        resp.raise_for_status()
        elapsed = time.perf_counter() - started_at
        data = resp.json()
        raw = str(data.get("response", ""))
        thinking = data.get("thinking", "")
        thinking_text = str(thinking) if thinking is not None else ""
        if not raw.strip() and thinking_text.strip():
            raw = thinking_text
        _debug_log(
            f"[stage3][{question_debug_id}][req#{req_counter}] RESPONSE in {elapsed:.2f}s | "
            f"keys={list(data.keys())} | response_len={len(raw)} | thinking_len={len(thinking_text)}\n"
            f"response_repr={raw!r}\n"
            f"response_text=\n{raw}"
        )
        if thinking_text:
            _debug_log(
                f"[stage3][{question_debug_id}][req#{req_counter}] THINKING\n{thinking_text}"
            )
        return raw

    def _parse_with_retries(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # Attempt 1: strict JSON mode.
        raw_local = _generate_raw(system_prompt, user_prompt, json_mode=True)
        try:
            return _extract_first_json(raw_local)
        except Exception:
            pass

        # Attempt 2: plain-text mode (some models fail in forced JSON mode).
        raw_local = _generate_raw(system_prompt, user_prompt, json_mode=False)
        try:
            return _extract_first_json(raw_local)
        except Exception:
            pass

        # Attempt 3: repair pass.
        repair_system = "You repair text into valid JSON. Return only JSON."
        repair_prompt = (
            "Convert the following content into a valid JSON object with keys "
            "\"answer\" and \"summary\" only.\n\n"
            f"{raw_local}"
        )
        repaired = _generate_raw(repair_system, repair_prompt, json_mode=False)
        return _extract_first_json(repaired)

    try:
        parsed = _parse_with_retries(ANSWER_SYSTEM_PROMPT, prompt)
        answer = str(parsed.get("answer", "")).strip() or "Unknown"
        summary = str(parsed.get("summary", "")).strip() or "No summary generated."
    except Exception:
        fallback_prompt = (
            prompt
            + "\n\nIf you cannot return JSON, output EXACTLY two lines:\n"
            + "ANSWER: <final answer>\nSUMMARY: <brief reasoning>"
        )
        fallback_raw = _generate_raw(ANSWER_SYSTEM_PROMPT, fallback_prompt, json_mode=False)
        parsed_text = _parse_answer_summary_text(fallback_raw)
        if parsed_text is None:
            question = question_payload.get("question", {}) or {}
            simple_prompt = (
                "Solve the following math question. The text may contain LaTeX.\n"
                "Return EXACTLY two lines:\n"
                "ANSWER: <final answer>\n"
                "SUMMARY: <brief reasoning>\n\n"
                f"Question text: {question.get('text', '')}\n"
                f"Options: {json.dumps(question.get('options', []), ensure_ascii=False)}"
            )
            simple_raw = _generate_raw("You are a precise math solver.", simple_prompt, json_mode=False)
            parsed_text = _parse_answer_summary_text(simple_raw)
        if parsed_text is None:
            raise ValueError("Model returned empty/non-parseable answer output.")
        answer = parsed_text["answer"] or "Unknown"
        summary = parsed_text["summary"] or "No summary generated."

    if _looks_like_placeholder(answer, summary) or _is_weak_answer(answer, summary):
        strict_system = ANSWER_SYSTEM_PROMPT + (
            "\nCRITICAL: Do not output templates/placeholders. Compute using the actual provided question and options."
        )
        strict_prompt = (
            prompt
            + "\n\nThe previous output was a placeholder/template. Re-solve and return real computed answer + summary."
        )
        parsed = _parse_with_retries(strict_system, strict_prompt)
        answer = str(parsed.get("answer", "")).strip() or "Unknown"
        summary = str(parsed.get("summary", "")).strip() or "No summary generated."

    # Final MCQ rescue: force one option selection if answer is still weak.
    if is_mcq and ((answer or "").strip().lower() in {"", "unknown"}):
        mcq_prompt = (
            "You must solve this MCQ and choose exactly one option.\n"
            "Return EXACTLY two lines:\n"
            "ANSWER: <exact chosen option text>\n"
            "SUMMARY: <brief reasoning with key steps>\n\n"
            f"Question text: {qtext}\n"
            f"Options: {json.dumps(options, ensure_ascii=False)}"
        )
        mcq_raw = _generate_raw("You are a strict mathematics MCQ solver.", mcq_prompt, json_mode=False)
        parsed_text = _parse_answer_summary_text(mcq_raw)
        if parsed_text is not None:
            answer = (parsed_text.get("answer") or "").strip() or answer
            summary = (parsed_text.get("summary") or "").strip() or summary

    if is_mcq:
        mapped = _normalize_mcq_answer_to_index(answer, [str(x) for x in options])
        if mapped:
            answer = mapped

    if not summary.strip():
        summary = "No summary generated."

    return {"answer": answer, "summary": summary}


def _collect_question_refs(doc: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    refs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for section in doc.get("sections", []) or []:
        for question in section.get("questions", []) or []:
            refs.append((section, question))
    return refs


def _build_solver_payload(metadata: Dict[str, Any], section: Dict[str, Any], question: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = str(question.get("text", "") or "")
    text_core = raw_text.split("## **Explanation:**", 1)[0]
    text_core = re.sub(r"\s+", " ", text_core).strip()
    text_core = re.sub(r"^#+\s*", "", text_core)

    options = question.get("options", []) or []
    if not options:
        inline_opts = re.findall(r"\(([A-Da-d])\)\s*([^()]+?)(?=\s*\([A-Da-d]\)|$)", text_core)
        options = [opt_text.strip(" -") for _, opt_text in inline_opts if opt_text.strip(" -")]

    q_type = str(question.get("type", "Generic")).upper()
    if q_type in {"MCQ", "ASSERTION-REASON"} and options:
        # For MCQs, send only text + options.
        return {
            "question": {
                "text": text_core,
                "options": options,
            }
        }

    # For non-MCQ, send only question text.
    return {"question": {"text": text_core}}


def enrich_json_with_answers(
    extracted_json: Dict[str, Any],
    config: Optional[OllamaAnswerConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    config = config or OllamaAnswerConfig()
    enriched = copy.deepcopy(extracted_json)
    metadata = enriched.get("metadata", {}) or {}
    refs = _collect_question_refs(enriched)

    total = len(refs)
    if total == 0:
        return enriched

    for idx, (section, question) in enumerate(refs, start=1):
        q_num = question.get("question_number", idx)
        detail = f"q{q_num}"
        if progress_callback is not None:
            progress_callback(idx - 1, total, detail)

        payload = _build_solver_payload(metadata, section, question)
        payload["_debug"] = debug
        payload["_debug_question_id"] = detail
        question_started = time.perf_counter()
        try:
            answer_obj = _request_ollama_answer(config, payload)
        except Exception as exc:
            answer_obj = {
                "answer": "Unknown",
                "summary": (
                    f"Could not generate answer: {type(exc).__name__}: {str(exc)} "
                    f"(url={config.base_url.rstrip('/')}, model={config.model})"
                ),
            }
        if debug:
            question_elapsed = time.perf_counter() - question_started
            print(
                f"[stage3][{detail}] FINAL in {question_elapsed:.2f}s | answer={answer_obj.get('answer')}",
                file=sys.stderr,
                flush=True,
            )
        question["answer_section"] = answer_obj

        if progress_callback is not None:
            progress_callback(idx, total, detail)

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Add answers to extracted JSON.")
    parser.add_argument("input_json_path", help="Path to stage-2 JSON file")
    parser.add_argument("output_json_path", help="Path to enriched JSON output file")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--num-ctx", type=int, default=32768, help="Context window")
    parser.add_argument("--num-predict", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--debug", action="store_true", help="Print request/response debug logs per question")
    args = parser.parse_args()

    in_path = Path(args.input_json_path).expanduser().resolve()
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    out_path = Path(args.output_json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = OllamaAnswerConfig(
        base_url=args.ollama_url,
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    enriched = enrich_json_with_answers(raw, config=config, debug=args.debug)
    out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
