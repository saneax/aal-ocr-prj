#!/usr/bin/env python3
"""End-to-end OCR pipeline for math question papers.

Stage 1: PDF -> Markdown + extracted images (marker-pdf)
Stage 2: Markdown -> structured JSON
Stage 3: JSON -> answered JSON with summary (local Ollama)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from PIL import Image

from stage1_marker import MarkerRunConfig, Stage1Result, convert_pdf_to_markdown
from stage2_ollama import OllamaConfig, extract_json_from_markdown
from stage3_ollama import OllamaAnswerConfig, enrich_json_with_answers


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _render_progress(current: int, total: int, label: str, detail: str = "") -> None:
    total = max(total, 1)
    current = min(max(current, 0), total)
    width = 34
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(ratio * 100)
    message = f"[{bar}] {pct:3d}% ({current}/{total}) {label}"
    if detail:
        message += f" | {detail}"
    print(f"\r{message}", end="", file=sys.stderr, flush=True)
    if current == total:
        print(file=sys.stderr)


def _display_name(path: Path) -> str:
    return path.name or str(path)


class PipelineProgress:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.completed = 0

    def start(self, label: str, detail: str = "") -> None:
        _render_progress(self.completed, self.total_steps, label, detail)

    def advance(self, label: str, detail: str = "") -> None:
        self.completed += 1
        _render_progress(self.completed, self.total_steps, label, detail)


def infer_series(input_path: Path, explicit_series: str | None) -> str:
    if explicit_series:
        return explicit_series
    stem = input_path.stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def _normalize_ollama_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("Ollama URL cannot be empty.")
    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Invalid Ollama URL: {raw_url}")

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"Invalid Ollama URL: {raw_url}")

    port = parsed.port or 11434
    netloc = f"{hostname}:{port}"
    normalized = urlunparse((parsed.scheme, netloc, "", "", "", ""))
    return normalized.rstrip("/")


def _wipe_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_image_paths(image_paths: list[Path]) -> list[Path]:
    resolved_paths: list[Path] = []
    for image_path in image_paths:
        resolved = image_path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Invalid image path: {resolved}")
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension for: {resolved}")
        resolved_paths.append(resolved)
    return resolved_paths


def _build_generated_pdf_path(image_paths: list[Path]) -> Path:
    if not image_paths:
        raise ValueError("No image paths were provided.")

    first_stem = image_paths[0].stem.strip().replace(" ", "-") or "images"
    total_images = len(image_paths)
    file_name = f"{first_stem}-({total_images}).pdf"

    project_root = Path(__file__).resolve().parent
    input_pdf_dir = project_root / "input-pdf"
    input_pdf_dir.mkdir(parents=True, exist_ok=True)
    return input_pdf_dir / file_name


def build_pdf_from_images(image_paths: list[Path], target_pdf_path: Path) -> Path:
    """Create a single PDF from one or more image paths."""
    pil_images: list[Image.Image] = []
    total_steps = len(image_paths) + 1
    _render_progress(0, total_steps, "Preparing images", f"target={target_pdf_path.name}")
    try:
        for index, image_path in enumerate(image_paths, start=1):
            pil_images.append(Image.open(image_path).convert("RGB"))
            _render_progress(index, total_steps, "Preparing images", image_path.name)

        first_image, rest_images = pil_images[0], pil_images[1:]
        first_image.save(target_pdf_path, save_all=True, append_images=rest_images)
        _render_progress(total_steps, total_steps, "Image PDF ready", target_pdf_path.name)
    finally:
        for image in pil_images:
            image.close()

    return target_pdf_path


def _stage_paths(output_dir: Path, series: str) -> tuple[Path, Path, Path, Path]:
    markdown_path = output_dir / f"{series}.md"
    images_dir = output_dir / f"{series}_images"
    json_path = output_dir / f"{series}.json"
    answer_json_path = output_dir / f"{series}_with_answer.json"
    return markdown_path, images_dir, json_path, answer_json_path


def _build_pdf_input(args: argparse.Namespace, stage1_needed: bool) -> tuple[Path | None, Path]:
    if args.pdf_path:
        pdf_path = Path(args.pdf_path).expanduser().resolve()
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")
        return pdf_path, pdf_path

    image_paths = _validate_image_paths([Path(path) for path in (args.image_path or [])])
    if stage1_needed:
        pdf_path = build_pdf_from_images(image_paths, _build_generated_pdf_path(image_paths))
    else:
        pdf_path = None
    return pdf_path, image_paths[0]


def _run_stage1(
    pdf_path: Path,
    output_dir: Path,
    series: str,
    marker_prefer_gpu: bool,
    marker_low_vram_mode: bool,
    markdown_path: Path,
    images_dir: Path,
    reuse_existing: bool,
    progress: PipelineProgress,
) -> Stage1Result:
    if reuse_existing and markdown_path.exists():
        markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
        progress.advance("Stage 1 reused", markdown_path.name)
        return Stage1Result(markdown_text=markdown_text, markdown_path=markdown_path, images_dir=images_dir)

    progress.start("Stage 1", pdf_path.name)
    marker_cfg = MarkerRunConfig(prefer_gpu=marker_prefer_gpu, low_vram_mode=marker_low_vram_mode)
    stage1_result = convert_pdf_to_markdown(
        pdf_path=pdf_path,
        output_dir=output_dir,
        series=series,
        marker_config=marker_cfg,
    )
    progress.advance("Stage 1 complete", stage1_result.markdown_path.name)
    return stage1_result


def _run_stage2(
    stage1_result: Stage1Result,
    series: str,
    ollama_url: str,
    stage2_model: str,
    num_ctx: int,
    num_predict: int,
    max_chunk_chars: int,
    json_path: Path,
    reuse_existing: bool,
    progress: PipelineProgress,
) -> dict:
    if reuse_existing and json_path.exists():
        final_json = json.loads(json_path.read_text(encoding="utf-8"))
        progress.advance("Stage 2 reused", json_path.name)
        progress.advance("Stage 2 JSON reused", json_path.name)
        return final_json

    progress.start("Stage 2", stage2_model)
    ollama_config = OllamaConfig(
        base_url=ollama_url,
        model=stage2_model,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    final_json = extract_json_from_markdown(
        markdown_text=stage1_result.markdown_text,
        series=series,
        config=ollama_config,
        max_chunk_chars=max_chunk_chars,
        markdown_path=stage1_result.markdown_path,
    )
    progress.advance("Stage 2 complete", f"sections={len(final_json.get('sections', []))}")

    progress.start("Writing JSON", json_path.name)
    json_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.advance("Stage 2 JSON ready", json_path.name)
    return final_json


def _run_stage3(
    final_json: dict,
    answer_json_path: Path,
    ollama_url: str,
    stage3_model: str,
    num_ctx: int,
    num_predict: int,
    reuse_existing: bool,
    progress: PipelineProgress,
    debug: bool = False,
) -> dict:
    if reuse_existing and answer_json_path.exists():
        answered_json = json.loads(answer_json_path.read_text(encoding="utf-8"))
        progress.advance("Stage 3 reused", answer_json_path.name)
        progress.advance("Done (reused)", answer_json_path.name)
        return answered_json

    progress.start("Stage 3", "answer generation")
    answer_config = OllamaAnswerConfig(
        base_url=ollama_url,
        model=stage3_model,
        num_ctx=num_ctx,
        num_predict=min(num_predict, 2048),
    )

    question_start_times: dict[str, float] = {}
    durations: list[float] = []

    def _stage3_progress(current: int, total: int, detail: str) -> None:
        # Called twice per question: once before solving (current=idx-1), once after (current=idx).
        if detail not in question_start_times:
            question_start_times[detail] = time.perf_counter()
            _render_progress(current, total, "Answering questions", f"{detail} ...")
            return

        elapsed = time.perf_counter() - question_start_times.pop(detail)
        durations.append(elapsed)
        avg = sum(durations) / len(durations)
        remaining = max(total - current, 0)
        eta = avg * remaining
        _render_progress(
            current,
            total,
            "Answering questions",
            f"{detail} {elapsed:.1f}s | avg {avg:.1f}s | eta {eta:.0f}s",
        )

    answered_json = enrich_json_with_answers(
        extracted_json=final_json,
        config=answer_config,
        progress_callback=_stage3_progress,
        debug=debug,
    )
    total_questions = sum(len(sec.get("questions", []) or []) for sec in answered_json.get("sections", []) or [])
    progress.advance("Stage 3 complete", f"questions={total_questions}")

    progress.start("Writing answers", answer_json_path.name)
    answer_json_path.write_text(json.dumps(answered_json, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.advance("Done", answer_json_path.name)
    return answered_json


def _progress_steps_for_stage(stage: int) -> int:
    if stage == 1:
        return 1
    if stage == 2:
        return 3
    return 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Math paper OCR pipeline")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf-path", default=None, help="Input PDF file")
    input_group.add_argument(
        "--image-path",
        nargs="+",
        default=None,
        help="One or more input images (.png/.jpg/.jpeg) to merge into a single PDF",
    )

    parser.add_argument("--output-dir", required=True, help="Directory to write markdown/images/json")
    parser.add_argument("--series", default=None, help="Paper series identifier (optional)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3, help="Pipeline stage to run up to")
    parser.add_argument("--ollama-url", default="http://192.168.2.156:11434", help="Ollama base URL")
    parser.add_argument("--stage2-model", default="llama3.2:3b", help="Ollama model name for Stage 2")
    parser.add_argument("--stage3-model", default="llama3.2:3b", help="Ollama model name for Stage 3")
    parser.add_argument("--model", default=None, help="Deprecated fallback model (applies to both Stage 2 and Stage 3)")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run marker on CPU only (more stable, slower).",
    )
    parser.add_argument(
        "--disable-low-vram-mode",
        action="store_true",
        help="Disable conservative low-VRAM marker settings.",
    )
    parser.add_argument("--num-ctx", type=int, default=32768, help="Ollama context window")
    parser.add_argument("--num-predict", type=int, default=8192, help="Max generated tokens")
    parser.add_argument("--max-chunk-chars", type=int, default=12000, help="Chunk size fallback for long markdown")
    parser.add_argument("--wipe", action="store_true", help="Wipe output directory before running the pipeline")
    parser.add_argument("--debug", action="store_true", help="Print Stage 3 request/response debug logs")
    args = parser.parse_args()

    ollama_url = _normalize_ollama_url(args.ollama_url)
    stage2_model = args.stage2_model
    stage3_model = args.stage3_model
    if args.model:
        stage2_model = args.model
        stage3_model = args.model
    output_dir = Path(args.output_dir).expanduser().resolve()

    progress = PipelineProgress(total_steps=_progress_steps_for_stage(args.stage))
    if args.wipe:
        progress.start("Wiping outputs", _display_name(output_dir))
        _wipe_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine series first from direct input path (PDF or first image), without requiring Stage 1 execution.
    if args.pdf_path:
        series_input_path = Path(args.pdf_path).expanduser().resolve()
    else:
        image_paths = _validate_image_paths([Path(path) for path in (args.image_path or [])])
        series_input_path = image_paths[0]

    series = infer_series(series_input_path, args.series)
    markdown_path, images_dir, json_path, answer_json_path = _stage_paths(output_dir, series)

    reuse_allowed = not args.wipe

    stage1_result: Stage1Result | None = None
    final_json: dict | None = None
    answered_json: dict | None = None

    # Stage 1 execution is needed only when we cannot reuse stage1/stage2 artifacts.
    stage1_needed = args.stage == 1 or not (reuse_allowed and (markdown_path.exists() or json_path.exists() or answer_json_path.exists()))
    pdf_path, _ = _build_pdf_input(args, stage1_needed=stage1_needed)

    if args.stage >= 1:
        if args.stage == 1:
            if pdf_path is None:
                raise RuntimeError("Stage 1 requested but no PDF input could be prepared.")
            stage1_result = _run_stage1(
                pdf_path=pdf_path,
                output_dir=output_dir,
                series=series,
                marker_prefer_gpu=not args.cpu_only,
                marker_low_vram_mode=not args.disable_low_vram_mode,
                markdown_path=markdown_path,
                images_dir=images_dir,
                reuse_existing=reuse_allowed,
                progress=progress,
            )
        elif reuse_allowed and json_path.exists():
            progress.advance("Stage 1 skipped", "Using existing Stage 2 JSON")
        elif reuse_allowed and markdown_path.exists():
            if pdf_path is None:
                raise RuntimeError("Stage 1 markdown exists but PDF input is unavailable for consistency checks.")
            stage1_result = _run_stage1(
                pdf_path=pdf_path,
                output_dir=output_dir,
                series=series,
                marker_prefer_gpu=not args.cpu_only,
                marker_low_vram_mode=not args.disable_low_vram_mode,
                markdown_path=markdown_path,
                images_dir=images_dir,
                reuse_existing=True,
                progress=progress,
            )
        else:
            if pdf_path is None:
                raise RuntimeError("Upstream artifacts missing; unable to prepare PDF input for Stage 1.")
            stage1_result = _run_stage1(
                pdf_path=pdf_path,
                output_dir=output_dir,
                series=series,
                marker_prefer_gpu=not args.cpu_only,
                marker_low_vram_mode=not args.disable_low_vram_mode,
                markdown_path=markdown_path,
                images_dir=images_dir,
                reuse_existing=False,
                progress=progress,
            )

    if args.stage >= 2:
        if reuse_allowed and json_path.exists():
            # Stage 2 already complete; reuse unless explicitly wiped.
            final_json = json.loads(json_path.read_text(encoding="utf-8"))
            progress.advance("Stage 2 reused", json_path.name)
            progress.advance("Stage 2 JSON reused", json_path.name)
        else:
            if stage1_result is None:
                if markdown_path.exists() and reuse_allowed:
                    markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
                    stage1_result = Stage1Result(
                        markdown_text=markdown_text,
                        markdown_path=markdown_path,
                        images_dir=images_dir,
                    )
                    progress.advance("Stage 1 reused", markdown_path.name)
                else:
                    if pdf_path is None:
                        raise RuntimeError("Stage 2 requested but no Stage 1 artifacts or PDF input available.")
                    stage1_result = _run_stage1(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        series=series,
                        marker_prefer_gpu=not args.cpu_only,
                        marker_low_vram_mode=not args.disable_low_vram_mode,
                        markdown_path=markdown_path,
                        images_dir=images_dir,
                        reuse_existing=False,
                        progress=progress,
                    )

            final_json = _run_stage2(
                stage1_result=stage1_result,
                series=series,
                ollama_url=ollama_url,
                stage2_model=stage2_model,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                max_chunk_chars=args.max_chunk_chars,
                json_path=json_path,
                reuse_existing=False,
                progress=progress,
            )

    if args.stage >= 3:
        if final_json is None:
            if json_path.exists() and reuse_allowed:
                final_json = json.loads(json_path.read_text(encoding="utf-8"))
                progress.advance("Stage 1 skipped", "Using existing Stage 2 JSON")
                progress.advance("Stage 2 reused", json_path.name)
                progress.advance("Stage 2 JSON reused", json_path.name)
            else:
                if stage1_result is None:
                    if markdown_path.exists() and reuse_allowed:
                        markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
                        stage1_result = Stage1Result(
                            markdown_text=markdown_text,
                            markdown_path=markdown_path,
                            images_dir=images_dir,
                        )
                        progress.advance("Stage 1 reused", markdown_path.name)
                    else:
                        if pdf_path is None:
                            raise RuntimeError("Stage 3 requested but no reusable outputs and no PDF input available.")
                        stage1_result = _run_stage1(
                            pdf_path=pdf_path,
                            output_dir=output_dir,
                            series=series,
                            marker_prefer_gpu=not args.cpu_only,
                            marker_low_vram_mode=not args.disable_low_vram_mode,
                            markdown_path=markdown_path,
                            images_dir=images_dir,
                            reuse_existing=False,
                            progress=progress,
                        )
                final_json = _run_stage2(
                    stage1_result=stage1_result,
                    series=series,
                    ollama_url=ollama_url,
                    stage2_model=stage2_model,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    max_chunk_chars=args.max_chunk_chars,
                    json_path=json_path,
                    reuse_existing=False,
                    progress=progress,
                )

        answered_json = _run_stage3(
            final_json=final_json,
            answer_json_path=answer_json_path,
            ollama_url=ollama_url,
            stage3_model=stage3_model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            reuse_existing=reuse_allowed,
            progress=progress,
            debug=args.debug,
        )

    result = {
        "series": series,
        "markdown_path": str(markdown_path),
        "images_dir": str(images_dir),
        "json_path": str(json_path),
        "answer_json_path": str(answer_json_path),
        "stage": args.stage,
        "reused_outputs": reuse_allowed,
        "stage2_model": stage2_model,
        "stage3_model": stage3_model,
    }

    # Keep explicit stage completion details visible.
    if args.stage == 1:
        result["stage1_done"] = stage1_result is not None
    elif args.stage == 2:
        result["stage2_done"] = final_json is not None
    else:
        result["stage3_done"] = answered_json is not None

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
