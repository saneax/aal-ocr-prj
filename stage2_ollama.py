#!/usr/bin/env python3
"""Stage 2: Convert Markdown to structured JSON using local Ollama."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image


SYSTEM_PROMPT_TEMPLATE = """
You are a highly accurate data extraction assistant. Your task is to extract data from the provided mathematics question paper (provided as Markdown) into a strict JSON file.

Apply the following strict constraints:
1. Language: Only extract English text. Ignore Hindi translations entirely.
2. Noise Filtering: Filter out all non-question parts, like general instructions, watermark text, or page footers.
3. Metadata: Extract the PDF metadata: 'Series,' 'Q.P. Code,' and 'Subject.' 
4. Images & Artifacts: The document contains Markdown image tags (e.g., `![image](path.png)`). Extract the file path of the barcode and place it in `barcode_image`. For questions with diagrams/graphs, extract the image path and place it in `artifact_link`.
5. Structure: Group questions by their section. Fetch the section instructions and tag them.
6. Question Tagging: Tag each question strictly as "MCQ", "Assertion-Reason", or "Generic".
7. Options & Marks: If it's an MCQ or Assertion-Reason, pick up all options as sub-elements. Fill in the marks for all questions.
8. Math Formatting: Format all mathematical formulas, integrals, and matrices using standard LaTeX.
9. COMPLETENESS (CRITICAL): You must extract every single question. Do not truncate, skip questions, or provide a sample. The entire paper must be mapped.
10. SPARSE INPUT FALLBACK: If metadata is missing, fill defaults. If sections are missing, put all questions under "SECTION - A". If only one unnumbered question exists, still emit it as question_number 1.

The output MUST be a valid JSON object matching this exact schema:
{{
  "metadata": {{
    "series": "{series}",
    "qp_code": "",
    "subject": "",
    "maximum_marks": "",
    "time_allowed": "",
    "barcode_image": ""
  }},
  "sections": [
    {{
      "section_name": "",
      "instructions": "",
      "questions": [
        {{
          "question_number": 1,
          "type": "MCQ", 
          "marks": 1,
          "text": "Question text with LaTeX goes here",
          "artifact_link": "path/to/image.png",
          "options": ["Option A", "Option B"]
        }}
      ]
    }}
  ]
}}

If a question is not multiple-choice, omit the "options" array or leave it empty. If there is no artifact, omit the "artifact_link" key.
If metadata fields are missing, use: qp_code="UNKNOWN", subject="Unknown", maximum_marks="0", time_allowed="Unknown", barcode_image="".
If no section markers exist, output one section with section_name="SECTION - A" and instructions="No section heading found; defaulted to Section A.".
Provide ONLY the raw JSON object. Do not include introductory text, explanations, or markdown wrappers like ```json.
""".strip()


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b"
    timeout_sec: int = 600
    num_ctx: int = 32768
    num_predict: int = 8192
    temperature: float = 0.0


def _contains_devanagari(text: str) -> bool:
    return re.search(r"[\u0900-\u097F]", text) is not None


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _extract_metadata_from_markdown(markdown_text: str, series: str) -> Dict[str, str]:
    qp_code = ""
    subject = ""
    maximum_marks = ""
    time_allowed = ""
    barcode_image = ""

    m = re.search(r"Q\.P\.\s*Code\s*([A-Za-z0-9/.-]+)", markdown_text, flags=re.IGNORECASE)
    if m:
        qp_code = m.group(1).strip()

    subj_match = re.search(r"(?m)^\s*##\s*\*{0,2}\s*MATHEMATICS\s*\*{0,2}\s*$", markdown_text, flags=re.IGNORECASE)
    if subj_match:
        subject = "Mathematics"

    m = re.search(r"Maximum\s+Marks\s*:?\s*([0-9]+)", markdown_text, flags=re.IGNORECASE)
    if m:
        maximum_marks = m.group(1).strip()

    m = re.search(r"Time\s+allowed\s*:?\s*([0-9 ]+hours?)", markdown_text, flags=re.IGNORECASE)
    if m:
        time_allowed = _clean_line(m.group(1))

    img_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown_text)
    if img_refs:
        barcode_image = img_refs[0]

    return {
        "series": series,
        "qp_code": qp_code,
        "subject": subject,
        "maximum_marks": maximum_marks,
        "time_allowed": time_allowed,
        "barcode_image": barcode_image,
    }


def _section_marks(section_letter: str) -> int:
    return {"A": 1, "B": 2, "C": 3, "D": 5, "E": 4}.get(section_letter, 0)


def _section_name(section_letter: str) -> str:
    return f"SECTION - {section_letter}"


def _section_instructions(section_letter: str) -> str:
    mapping = {
        "A": "20 MCQ/Assertion-Reason questions of 1 mark each.",
        "B": "5 Very Short Answer (VSA) questions of 2 marks each.",
        "C": "6 Short Answer (SA) questions of 3 marks each.",
        "D": "4 Long Answer (LA) questions of 5 marks each.",
        "E": "3 Case study based questions of 4 marks each.",
    }
    return mapping.get(section_letter, "")


def _question_type(qnum: int, has_options: bool) -> str:
    if 1 <= qnum <= 18 or has_options:
        return "MCQ"
    if 19 <= qnum <= 20:
        return "Assertion-Reason"
    return "Generic"


def _compose_images(image_paths: List[Path], out_path: Path) -> Optional[Path]:
    valid = [p for p in image_paths if p.exists()]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    images: List[Image.Image] = []
    for p in valid:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not images:
        return None
    if len(images) == 1:
        return valid[0]

    width = max(img.width for img in images)
    pad = 12
    total_height = sum(img.height for img in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    y = 0
    for img in images:
        x = (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _parse_questions_deterministically(markdown_text: str, md_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    lines = markdown_text.splitlines()

    # Parse only English main body starting at Section A heading.
    start_idx = 0
    for i, line in enumerate(lines):
        if re.search(r"\bSECTION\s*-\s*A\b", line, flags=re.IGNORECASE):
            start_idx = i
            break
    lines = lines[start_idx:]

    filtered: List[str] = []
    for ln in lines:
        if ln.strip().startswith("!["):
            filtered.append(ln)
            continue
        if _contains_devanagari(ln):
            continue
        if re.search(r"\bPage\s+\d+\s+of\s+\d+\b", ln):
            continue
        if re.fullmatch(r"\s*65/2/2\s*", ln.strip()):
            continue
        filtered.append(ln)

    q_start = re.compile(r"^\s*[-*]?\s*(\d{1,2})\.\s*(.*)$")
    sec_header = re.compile(r"^\s*#*\s*SECTION\s*-\s*([A-E])\b", flags=re.IGNORECASE)
    opt_re = re.compile(r"^\s*[-*]?\s*\(?([A-D])\)\s*(.*)$")
    inline_opt_re = re.compile(r"\(([A-D])\)\s*")
    img_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

    current_section = "A"
    pending: Optional[Dict[str, Any]] = None
    parsed: List[Dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return

        body_lines = [ln for ln in pending["lines"] if ln.strip()]
        options: List[tuple[str, str]] = []
        q_text_lines: List[str] = []
        current_opt_idx: Optional[int] = None
        for ln in body_lines:
            # Parse inline "(A) ... (B) ..." options when OCR keeps multiple options on one line.
            inline_matches = list(inline_opt_re.finditer(ln))
            if inline_matches and not ln.strip().startswith("!["):
                consumed = False
                if len(inline_matches) >= 1:
                    for i, m_opt in enumerate(inline_matches):
                        start = m_opt.end()
                        end = inline_matches[i + 1].start() if i + 1 < len(inline_matches) else len(ln)
                        opt_text = _clean_line(ln[start:end])
                        if opt_text:
                            options.append((m_opt.group(1).upper(), opt_text))
                            consumed = True
                    if consumed:
                        current_opt_idx = len(options) - 1
                        continue

            opt_m = opt_re.match(ln)
            if opt_m:
                current_opt_idx = len(options)
                opt_text = _clean_line(opt_m.group(2))
                options.append((opt_m.group(1).upper(), opt_text))
                continue
            if current_opt_idx is not None and ln.strip() and not ln.strip().startswith("![]("):
                # Continuation for previous option.
                label, prev_text = options[current_opt_idx]
                options[current_opt_idx] = (label, _clean_line(f"{prev_text} {ln}"))
                continue
            if not ln.strip().startswith("![]("):
                q_text_lines.append(ln)

        q_text = _clean_line(" ".join(q_text_lines))
        qnum = pending["question_number"]
        has_options = any(_clean_line(text) for _, text in options)
        qtype = _question_type(qnum, has_options=has_options)
        marks = _section_marks(pending["section"])

        image_refs = []
        for ln in body_lines:
            image_refs.extend(img_re.findall(ln))

        artifact_link = None
        if image_refs and md_path is not None:
            md_dir = md_path.parent
            img_abs = [md_dir / ref for ref in image_refs]
            composite_out = md_dir / "composite_artifacts" / f"q{qnum}_composite.jpg"
            composed = _compose_images(img_abs, composite_out)
            if composed is not None:
                artifact_link = str(composed.relative_to(md_dir))
        elif image_refs:
            artifact_link = image_refs[0]

        obj: Dict[str, Any] = {
            "question_number": qnum,
            "type": qtype,
            "marks": marks,
            "text": q_text,
        }
        if options:
            # Keep only non-empty options and cap at 4 choices for MCQ consistency.
            cleaned_opts = []
            for label, text in options:
                cleaned = _clean_line(text)
                if cleaned:
                    cleaned_opts.append(f"({label}) {cleaned}")
            if cleaned_opts:
                obj["options"] = cleaned_opts[:4]
        if artifact_link:
            obj["artifact_link"] = artifact_link
        parsed.append(obj)
        pending = None

    for raw in filtered:
        line = raw.rstrip()
        sec_m = sec_header.match(line)
        if sec_m:
            flush_pending()
            current_section = sec_m.group(1).upper()
            continue

        m = q_start.match(line)
        if m:
            qn = int(m.group(1))
            if 1 <= qn <= 38:
                flush_pending()
                pending = {
                    "question_number": qn,
                    "section": current_section,
                    "lines": [m.group(2)],
                }
                continue

        if pending is not None:
            pending["lines"].append(line)
    flush_pending()

    # Deduplicate by question number; keep the longer text variant if duplicates appear.
    best: Dict[int, Dict[str, Any]] = {}
    for q in parsed:
        qn = q["question_number"]
        if qn not in best:
            best[qn] = q
            continue
        if len(q.get("text", "")) > len(best[qn].get("text", "")):
            best[qn] = q

    ordered = [best[n] for n in sorted(best.keys())]
    return ordered


def _build_sections_from_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranges = [
        ("A", 1, 20),
        ("B", 21, 25),
        ("C", 26, 31),
        ("D", 32, 35),
        ("E", 36, 38),
    ]
    sections: List[Dict[str, Any]] = []
    assigned_numbers: set[int] = set()
    for sec, start, end in ranges:
        qs = [q for q in questions if start <= q["question_number"] <= end]
        if not qs:
            continue
        assigned_numbers.update(q["question_number"] for q in qs)
        sections.append(
            {
                "section_name": _section_name(sec),
                "instructions": _section_instructions(sec),
                "questions": qs,
            }
        )
    remaining = [q for q in questions if q.get("question_number") not in assigned_numbers]
    if remaining:
        default_instructions = "No section heading found; defaulted to Section A."
        existing_a = next((s for s in sections if s.get("section_name") == _section_name("A")), None)
        if existing_a is not None:
            existing_a["questions"].extend(remaining)
        else:
            sections.append(
                {
                    "section_name": _section_name("A"),
                    "instructions": default_instructions,
                    "questions": remaining,
                }
            )
    return sections


def _fallback_single_question(markdown_text: str, md_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    lines = [ln.strip() for ln in markdown_text.splitlines() if ln.strip()]
    if not lines:
        return None

    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown_text)
    text_lines = [ln for ln in lines if not ln.startswith("![")]
    text = _clean_line(" ".join(text_lines))
    if not text:
        return None

    question: Dict[str, Any] = {
        "question_number": 1,
        "type": "Generic",
        "marks": 1,
        "text": text,
    }

    if image_refs and md_path is not None:
        md_dir = md_path.parent
        img_abs = [md_dir / ref for ref in image_refs]
        composite_out = md_dir / "composite_artifacts" / "q1_composite.jpg"
        composed = _compose_images(img_abs, composite_out)
        if composed is not None:
            question["artifact_link"] = str(composed.relative_to(md_dir))
    elif image_refs:
        question["artifact_link"] = image_refs[0]

    return question


def _ensure_defaults(obj: Dict[str, Any], series: str, markdown_text: str, md_path: Optional[Path]) -> Dict[str, Any]:
    metadata = obj.setdefault("metadata", {})
    metadata.setdefault("series", series)
    metadata["series"] = series

    defaults = {
        "qp_code": "UNKNOWN",
        "subject": "Unknown",
        "maximum_marks": "0",
        "time_allowed": "Unknown",
        "barcode_image": "",
    }
    for key, default_val in defaults.items():
        value = metadata.get(key)
        metadata[key] = value if value not in (None, "") else default_val

    sections = obj.get("sections")
    if not isinstance(sections, list):
        sections = []
    obj["sections"] = sections

    if not sections:
        fallback_question = _fallback_single_question(markdown_text, md_path)
        questions = [fallback_question] if fallback_question is not None else []
        obj["sections"] = [
            {
                "section_name": "SECTION - A",
                "instructions": "No section heading found; defaulted to Section A.",
                "questions": questions,
            }
        ]
    return obj


def _extract_first_json(raw_text: str) -> Dict[str, Any]:
    """Strictly parse JSON, with a small fallback to recover fenced/raw wrappers."""
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first top-level JSON object in case model emitted extra text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a valid JSON object.")

    candidate = text[start : end + 1]
    return json.loads(candidate)


def _smart_markdown_chunks(markdown_text: str, max_chars: int = 12000) -> List[str]:
    """Chunk markdown by section headings while staying under max_chars."""
    if len(markdown_text) <= max_chars:
        return [markdown_text]

    # Split on markdown headings while keeping them attached.
    pieces = re.split(r"(?=^#{1,6}\s)", markdown_text, flags=re.MULTILINE)
    chunks: List[str] = []
    current = ""

    for piece in pieces:
        if not piece.strip():
            continue
        if len(piece) > max_chars:
            # Hard split very large sections.
            for i in range(0, len(piece), max_chars):
                sub = piece[i : i + max_chars]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(sub)
            continue

        if len(current) + len(piece) <= max_chars:
            current += piece
        else:
            if current:
                chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    return chunks


def _request_ollama_json(config: OllamaConfig, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = f"{config.base_url.rstrip('/')}/api/generate"

    def _generate_raw(system: str, prompt: str, json_mode: bool = True) -> str:
        payload = {
            "model": config.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"
        resp = requests.post(url, json=payload, timeout=config.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "")
        # Some models return empty strings in JSON mode; retry as plain text.
        if json_mode and not raw:
            payload.pop("format", None)
            resp = requests.post(url, json=payload, timeout=config.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
        return raw

    # Primary attempt.
    raw = _generate_raw(system_prompt, user_prompt)
    try:
        return _extract_first_json(raw)
    except Exception:
        pass

    # Retry with stricter instruction.
    strict_prompt = (
        "CRITICAL: Return ONLY one valid JSON object. No commentary, no markdown, no code fences.\n\n"
        + user_prompt
    )
    raw_retry = _generate_raw(system_prompt, strict_prompt)
    try:
        return _extract_first_json(raw_retry)
    except Exception:
        pass

    # Final pass: ask model to repair prior output into strict JSON.
    repair_system = "You are a strict JSON repair assistant. Output only valid JSON."
    repair_prompt = (
        "Fix the following content into one valid JSON object. "
        "Keep all original extracted information. Output only raw JSON.\n\n"
        f"{raw_retry or raw}"
    )
    repaired = _generate_raw(repair_system, repair_prompt)
    try:
        return _extract_first_json(repaired)
    except Exception as exc:
        preview = (repaired or raw_retry or raw)[:400]
        raise ValueError(f"Model output is not valid JSON after retries. Preview: {preview}") from exc


def _normalize_question_numbers(sections: Sequence[Dict[str, Any]]) -> None:
    """Best-effort normalization of question_number field to ints where possible."""
    for section in sections:
        for q in section.get("questions", []) or []:
            val = q.get("question_number")
            if isinstance(val, int):
                continue
            if isinstance(val, str):
                m = re.search(r"\d+", val)
                if m:
                    q["question_number"] = int(m.group())


def _all_question_numbers(obj: Dict[str, Any]) -> set[int]:
    nums: set[int] = set()
    for section in obj.get("sections", []) or []:
        for q in section.get("questions", []) or []:
            qn = q.get("question_number")
            if isinstance(qn, int):
                nums.add(qn)
    return nums


def _infer_expected_question_count(markdown_text: str) -> int:
    # Strong signal from instruction pages like "contains 38 questions".
    matches = re.findall(r"contains\s+(\d+)\s+questions", markdown_text, flags=re.IGNORECASE)
    if matches:
        return max(int(x) for x in matches)

    # Fallback signal from numbered lines in markdown.
    nums = re.findall(r"(?m)^\s*[-*]?\s*(\d+)\s*[\).]", markdown_text)
    if nums:
        return max(int(x) for x in nums)
    return 0


def _merge_partial_outputs(partials: Sequence[Dict[str, Any]], series: str) -> Dict[str, Any]:
    """Merge chunk-level JSON outputs into final schema."""
    final_obj: Dict[str, Any] = {
        "metadata": {
            "series": series,
            "qp_code": "",
            "subject": "",
            "maximum_marks": "",
            "time_allowed": "",
            "barcode_image": "",
        },
        "sections": [],
    }

    # Metadata: keep first non-empty value for each key.
    for part in partials:
        metadata = part.get("metadata", {}) or {}
        for key in ["qp_code", "subject", "maximum_marks", "time_allowed", "barcode_image"]:
            if not final_obj["metadata"].get(key) and metadata.get(key):
                final_obj["metadata"][key] = metadata[key]

    section_map: Dict[str, Dict[str, Any]] = {}
    seen_questions: set[Tuple[str, Any, str]] = set()

    for part in partials:
        for section in part.get("sections", []) or []:
            section_name = (section.get("section_name") or "").strip() or "Unlabeled Section"
            instructions = (section.get("instructions") or "").strip()

            if section_name not in section_map:
                section_map[section_name] = {
                    "section_name": section_name,
                    "instructions": instructions,
                    "questions": [],
                }
            elif not section_map[section_name].get("instructions") and instructions:
                section_map[section_name]["instructions"] = instructions

            for q in section.get("questions", []) or []:
                q_text = (q.get("text") or "").strip()
                q_num = q.get("question_number")
                dedupe_key = (section_name, q_num, q_text)
                if dedupe_key in seen_questions:
                    continue
                seen_questions.add(dedupe_key)
                section_map[section_name]["questions"].append(q)

    final_sections = list(section_map.values())
    _normalize_question_numbers(final_sections)

    # Keep questions sorted by question number when present.
    for section in final_sections:
        section["questions"] = sorted(
            section.get("questions", []),
            key=lambda q: (q.get("question_number") is None, q.get("question_number", 10**9)),
        )

    final_obj["sections"] = final_sections
    return final_obj


def extract_json_from_markdown(
    markdown_text: str,
    series: str,
    config: Optional[OllamaConfig] = None,
    max_chunk_chars: int = 12000,
    markdown_path: Optional[Path] = None,
    deterministic_only: bool = True,
) -> Dict[str, Any]:
    """Convert markdown to strict JSON using Ollama.

    Strategy:
    1) Try full markdown in one request with large num_ctx/num_predict.
    2) If model truncates/fails JSON parse, fall back to chunked extraction + merge.
    """
    # Deterministic extraction first for completeness and stability.
    questions = _parse_questions_deterministically(markdown_text, md_path=markdown_path)
    if not questions:
        fallback_question = _fallback_single_question(markdown_text, md_path=markdown_path)
        if fallback_question is not None:
            questions = [fallback_question]
    deterministic_obj = {
        "metadata": _extract_metadata_from_markdown(markdown_text, series=series),
        "sections": _build_sections_from_questions(questions),
    }
    deterministic_obj = _ensure_defaults(deterministic_obj, series=series, markdown_text=markdown_text, md_path=markdown_path)
    if deterministic_only:
        return deterministic_obj

    config = config or OllamaConfig()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(series=series)
    expected_questions = _infer_expected_question_count(markdown_text)

    user_prompt = f"Paper series: {series}\n\nMarkdown:\n{markdown_text}"
    try:
        full = _request_ollama_json(config, system_prompt=system_prompt, user_prompt=user_prompt)
        if isinstance(full, dict) and "sections" in full and "metadata" in full:
            # Ensure series value stays aligned to caller-provided id.
            full.setdefault("metadata", {})["series"] = series
            full_count = len(_all_question_numbers(full))
            # If output is suspiciously incomplete, force chunked fallback.
            if expected_questions and full_count < max(1, int(expected_questions * 0.9)):
                raise ValueError(
                    f"Full-pass extraction incomplete ({full_count}/{expected_questions}); switching to chunk fallback."
                )
            return _ensure_defaults(full, series=series, markdown_text=markdown_text, md_path=markdown_path)
    except Exception:
        pass

    chunks = _smart_markdown_chunks(markdown_text, max_chars=max_chunk_chars)
    partials: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_prompt = (
            f"Paper series: {series}\n"
            f"Chunk {idx}/{len(chunks)} of markdown (extract all questions present in this chunk):\n\n"
            f"{chunk}"
        )
        partial = _request_ollama_json(config, system_prompt=system_prompt, user_prompt=chunk_prompt)
        partials.append(partial)

    merged = _merge_partial_outputs(partials, series=series)

    # Targeted recovery pass for missing question numbers.
    if expected_questions:
        present = _all_question_numbers(merged)
        missing = sorted(set(range(1, expected_questions + 1)) - present)
        if missing:
            missing_prompt = (
                f"Paper series: {series}\n"
                f"Extract ONLY these missing question numbers: {missing}.\n"
                "Return the same required JSON schema.\n\n"
                f"Markdown:\n{markdown_text}"
            )
            try:
                recovered = _request_ollama_json(config, system_prompt=system_prompt, user_prompt=missing_prompt)
                merged = _merge_partial_outputs([merged, recovered], series=series)
            except Exception:
                pass

    merged = _ensure_defaults(merged, series=series, markdown_text=markdown_text, md_path=markdown_path)
    merged_nums = _all_question_numbers(merged)
    deterministic_nums = _all_question_numbers(deterministic_obj)
    if len(deterministic_nums) > len(merged_nums):
        return deterministic_obj
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Markdown -> JSON via Ollama")
    parser.add_argument("markdown_path", help="Path to markdown file")
    parser.add_argument("output_json_path", help="Path to output json")
    parser.add_argument("--series", required=True, help="Paper series identifier")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--num-ctx", type=int, default=32768, help="Context window")
    parser.add_argument("--num-predict", type=int, default=8192, help="Max tokens to generate")
    parser.add_argument("--max-chunk-chars", type=int, default=12000, help="Chunk size for fallback")
    parser.add_argument(
        "--use-ollama-refine",
        action="store_true",
        help="Use Ollama refinement on top of deterministic extraction.",
    )
    args = parser.parse_args()

    markdown_path = Path(args.markdown_path).expanduser().resolve()
    markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")

    config = OllamaConfig(
        base_url=args.ollama_url,
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    final_json = extract_json_from_markdown(
        markdown_text=markdown_text,
        series=args.series,
        config=config,
        max_chunk_chars=args.max_chunk_chars,
        markdown_path=markdown_path,
        deterministic_only=not args.use_ollama_refine,
    )

    out_path = Path(args.output_json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
