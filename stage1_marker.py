#!/usr/bin/env python3
"""Stage 1: Convert a PDF to Markdown and extract images using marker-pdf.

This module tries the marker Python API first, then falls back to CLI commands
for compatibility across marker versions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MIN_FAST_TEXT_CHARS = 800


@dataclass
class Stage1Result:
    """Result payload for Stage 1 conversion."""

    markdown_text: str
    markdown_path: Path
    images_dir: Path


@dataclass
class MarkerRunConfig:
    """Runtime controls for marker execution on constrained GPUs."""

    prefer_gpu: bool = True
    low_vram_mode: bool = True
    disable_multiprocessing: bool = True


def _infer_series(pdf_path: Path) -> str:
    """Infer series from filename prefix before first underscore.

    Example: "65-2-2_Mathematics.pdf" -> "65-2-2"
    """
    stem = pdf_path.stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def _save_extracted_images(images: Dict[str, object], images_dir: Path) -> None:
    """Persist marker API image objects to disk.

    The marker API can return a dict of PIL Image objects keyed by filename.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    for name, image_obj in images.items():
        file_name = Path(name).name
        target = images_dir / file_name
        # Most marker versions return PIL images with .save()
        if hasattr(image_obj, "save"):
            image_obj.save(target)
        else:
            # Fallback: write raw bytes if provided as bytes-like object
            if isinstance(image_obj, (bytes, bytearray)):
                target.write_bytes(bytes(image_obj))


def _try_marker_python_api(pdf_path: Path, output_dir: Path, series: str) -> Optional[Stage1Result]:
    """Try converting through marker Python API (version-tolerant)."""
    images_dir = output_dir / f"{series}_images"
    markdown_path = output_dir / f"{series}.md"

    try:
        # Common marker API pattern in recent releases.
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict  # type: ignore
        from marker.output import text_from_rendered  # type: ignore

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        markdown_text, _, images = text_from_rendered(rendered)
        markdown_text = _normalize_markdown_text(markdown_text)

        _save_extracted_images(images or {}, images_dir)
        markdown_path.write_text(markdown_text, encoding="utf-8")
        return Stage1Result(markdown_text=markdown_text, markdown_path=markdown_path, images_dir=images_dir)
    except Exception:
        # Continue to CLI fallback.
        return None


def _rewrite_markdown_image_paths(markdown_text: str, images_dir: Path) -> str:
    """Rewrite Markdown image links to point to local images_dir by filename.

    This keeps links stable even if marker wrote temp or nested paths.
    """

    def _replace(match: re.Match[str]) -> str:
        alt_text, image_path = match.group(1), match.group(2)
        file_name = Path(image_path).name
        new_path = (images_dir / file_name).as_posix()
        return f"![{alt_text}]({new_path})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown_text)


def _flatten_single_column_table(block_lines: list[str]) -> Optional[list[str]]:
    if len(block_lines) < 2:
        return None

    row_pattern = re.compile(r"^\|(.+)\|$")
    rows: list[str] = []
    for line in block_lines:
        m = row_pattern.match(line.strip())
        if not m:
            return None
        rows.append(m.group(1))

    separator = rows[1].replace(" ", "")
    if not separator or not set(separator) <= {"-", ":"}:
        return None

    # Only rewrite clearly broken single-column tables.
    if any("|" in row for row in rows):
        return None

    cleaned_rows = [re.sub(r"\s+", " ", row).strip() for row in (rows[0], *rows[2:])]
    cleaned_rows = [row for row in cleaned_rows if row]
    return cleaned_rows if cleaned_rows else None


def _normalize_markdown_text(markdown_text: str) -> str:
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    lines = [line.rstrip() for line in text.split("\n")]

    normalized: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.strip().startswith("|"):
            block_start = idx
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                idx += 1
            flattened = _flatten_single_column_table(lines[block_start:idx])
            if flattened is not None:
                normalized.extend(flattened)
                normalized.append("")
                continue
            normalized.extend(lines[block_start:idx])
            continue

        cleaned = re.sub(r"\s+", " ", line).strip() if line.strip() else ""
        normalized.append(cleaned)
        idx += 1

    joined = "\n".join(normalized)
    joined = re.sub(r"(?mi)\b[o0]uestion\b", "Question", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip() + "\n"


def _move_images_to_target(root_dir: Path, images_dir: Path) -> None:
    """Collect discovered image files under root_dir into images_dir."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            target = images_dir / path.name
            if path.resolve() == target.resolve():
                continue
            shutil.move(str(path), str(target))


def _extract_images_with_pdfimages(pdf_path: Path, images_dir: Path) -> None:
    pdfimages_bin = shutil.which("pdfimages")
    if not pdfimages_bin:
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    prefix = images_dir / "pdfimg"
    try:
        subprocess.run(
            [pdfimages_bin, "-all", str(pdf_path), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        # If image extraction fails, keep going with text-only markdown.
        return


def _fast_text_has_good_coverage(raw_text: str) -> bool:
    question_nums = [int(v) for v in re.findall(r"(?im)^\s*question\s+(\d{1,3})\b", raw_text)]
    if not question_nums:
        return True

    unique_nums = sorted(set(question_nums))
    if unique_nums[0] > 10 and len(unique_nums) >= 10:
        # Common failure mode: extracted only the tail pages (e.g., Q76+).
        return False
    return True


def _try_poppler_fast_path(pdf_path: Path, output_dir: Path, series: str, force: bool = False) -> Optional[Stage1Result]:
    pdftotext_bin = shutil.which("pdftotext")
    if not pdftotext_bin:
        return None

    images_dir = output_dir / f"{series}_images"
    markdown_path = output_dir / f"{series}.md"

    try:
        proc = subprocess.run(
            [pdftotext_bin, "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    raw_text = (proc.stdout or "").strip()
    if len(raw_text) < MIN_FAST_TEXT_CHARS:
        return None
    if not force and not _fast_text_has_good_coverage(raw_text):
        return None

    markdown_text = _normalize_markdown_text(raw_text)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return Stage1Result(markdown_text=markdown_text, markdown_path=markdown_path, images_dir=images_dir)


def _build_marker_base_cmd(marker_single_bin: str, pdf_path: Path, output_dir: Path, cfg: MarkerRunConfig) -> list[str]:
    cmd = [
        marker_single_bin,
        str(pdf_path),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "markdown",
    ]
    if cfg.disable_multiprocessing:
        cmd.append("--disable_multiprocessing")
    if cfg.low_vram_mode:
        # Conservative defaults for 8GB GPUs to avoid OOM during detection/recognition.
        cmd.extend(
            [
                "--layout_batch_size",
                "1",
                "--detection_batch_size",
                "1",
                "--ocr_error_batch_size",
                "1",
                "--recognition_batch_size",
                "8",
                "--equation_batch_size",
                "1",
                "--lowres_image_dpi",
                "72",
                "--highres_image_dpi",
                "144",
            ]
        )
    return cmd


def _try_marker_cli(pdf_path: Path, output_dir: Path, series: str, cfg: MarkerRunConfig) -> Stage1Result:
    """Run marker CLI as a fallback path.

    We try a few compatible command shapes because marker CLI options differ by
    version.
    """
    images_dir = output_dir / f"{series}_images"
    markdown_path = output_dir / f"{series}.md"

    python_bin_dir = Path(sys.prefix).resolve() / "bin"

    def _resolve_cmd(name: str) -> str:
        # Prefer the current virtualenv's bin path, then fallback to PATH lookup.
        in_venv = python_bin_dir / name
        if in_venv.exists():
            return str(in_venv)
        return shutil.which(name) or name

    marker_single_bin = _resolve_cmd("marker_single")
    marker_bin = _resolve_cmd("marker")
    base_cmd = _build_marker_base_cmd(marker_single_bin, pdf_path, output_dir, cfg)

    # Retry strategy:
    # 1) GPU low-VRAM profile.
    # 2) GPU + reduced DPI if OOM.
    # 3) CPU fallback for stability.
    run_plans: list[tuple[list[str], dict[str, str]]] = []
    default_env = dict(os.environ)
    default_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if cfg.prefer_gpu:
        run_plans.append((base_cmd, default_env))
        run_plans.append(
            (
                base_cmd
                + [
                    "--lowres_image_dpi",
                    "72",
                    "--highres_image_dpi",
                    "144",
                ],
                default_env,
            )
        )

    cpu_env = dict(default_env)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    run_plans.append((base_cmd, cpu_env))

    # Last-resort compatibility path for environments where marker_single fails.
    run_plans.append(([marker_bin, str(pdf_path), "--output_dir", str(output_dir)], default_env))

    last_error = None
    for cmd, env in run_plans:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            break
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").lower()
            stdout = (exc.stdout or "").lower()
            last_error = RuntimeError(
                f"Command failed ({' '.join(cmd)}): {exc}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
            )
            # If not an OOM-related failure, keep trying next fallback anyway.
            if "outofmemory" in stderr or "out of memory" in stderr or "outofmemory" in stdout:
                continue
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Unable to run marker CLI for PDF conversion: {last_error}")

    md_files = sorted(output_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not md_files:
        raise RuntimeError("Marker CLI finished but no Markdown file was generated.")

    source_md = md_files[0]
    markdown_text = source_md.read_text(encoding="utf-8", errors="ignore")

    _move_images_to_target(output_dir, images_dir)
    rewritten = _rewrite_markdown_image_paths(markdown_text, images_dir)
    rewritten = _normalize_markdown_text(rewritten)

    markdown_path.write_text(rewritten, encoding="utf-8")

    return Stage1Result(markdown_text=rewritten, markdown_path=markdown_path, images_dir=images_dir)


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    output_dir: str | Path,
    series: Optional[str] = None,
    marker_config: Optional[MarkerRunConfig] = None,
    use_python_api: bool = False,
    use_fast_text_path: bool = True,
    extract_images_in_fast_path: bool = False,
    force_fast_text_path: bool = False,
) -> Stage1Result:
    """Convert a PDF to Markdown and extract images.

    Args:
        pdf_path: Input PDF path.
        output_dir: Output directory for markdown + images.
        series: Optional explicit series identifier.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF path: {pdf_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    series = series or _infer_series(pdf_path)
    marker_config = marker_config or MarkerRunConfig()

    if use_fast_text_path:
        poppler_result = _try_poppler_fast_path(pdf_path, output_dir, series, force=force_fast_text_path)
        if poppler_result is not None:
            if extract_images_in_fast_path:
                _extract_images_with_pdfimages(pdf_path, poppler_result.images_dir)
            return poppler_result

    if use_python_api:
        api_result = _try_marker_python_api(pdf_path, output_dir, series)
        if api_result is not None:
            return api_result

    return _try_marker_cli(pdf_path, output_dir, series, marker_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: PDF -> Markdown + images")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--series", default=None, help="Paper series identifier")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force marker to run on CPU (safer but slower).",
    )
    parser.add_argument(
        "--disable-fast-text-path",
        action="store_true",
        help="Disable pdftotext fast path and force marker-based extraction.",
    )
    parser.add_argument(
        "--extract-images-fast-path",
        action="store_true",
        help="When fast text path is used, also extract PDF images via pdfimages.",
    )
    parser.add_argument(
        "--force-fast-text-path",
        action="store_true",
        help="Force fast text path even when question coverage looks incomplete.",
    )
    args = parser.parse_args()

    marker_cfg = MarkerRunConfig(prefer_gpu=not args.cpu_only, low_vram_mode=True)
    result = convert_pdf_to_markdown(
        args.pdf_path,
        args.output_dir,
        args.series,
        marker_cfg,
        use_fast_text_path=not args.disable_fast_text_path,
        extract_images_in_fast_path=args.extract_images_fast_path,
        force_fast_text_path=args.force_fast_text_path,
    )
    print(
        json.dumps(
            {
                "markdown_path": str(result.markdown_path),
                "images_dir": str(result.images_dir),
                "markdown_chars": len(result.markdown_text),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
