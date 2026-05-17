# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: End-to-end entrypoint for the OCR pipeline (PDF to Markdown/images to JSON).
- `stage1_marker.py`: Stage 1 conversion using `marker-pdf` (API first, CLI fallback).
- `stage2_ollama.py`: Stage 2 extraction and normalization using a local Ollama model.
- `input-pdf/`: Sample input PDFs for local runs.
- `output/`, `output-gpu-test/`, `output-final/`: Generated artifacts (`.md`, `.json`, extracted/composed images).
- `requirements.txt`: Python dependencies.

Keep generated outputs out of core logic changes; limit PR diffs to source files unless output updates are intentional.

## Build, Test, and Development Commands
- `make install`: Safe dependency setup via `scripts/install_deps.sh` (recommended).
- `python -m venv .venv && source .venv/bin/activate`: Create and activate local environment manually.
- `python -m pip install -r requirements.txt`: Install runtime dependencies manually.
- `python main.py input-pdf/65-2-2_Mathematics.pdf output-final`: Run full pipeline.
- `python stage1_marker.py --help`: Inspect Stage 1 options.
- `python stage2_ollama.py --help`: Inspect Stage 2 options.
- `python -m py_compile main.py stage1_marker.py stage2_ollama.py`: Quick syntax validation.
- `make clean-leaks`: Remove accidental files created by shell redirection mistakes.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear, typed function signatures.
- Use `snake_case` for functions/variables, `PascalCase` for dataclasses, and uppercase constants (for example, `SYSTEM_PROMPT_TEMPLATE`).
- Keep modules focused by stage responsibility; shared helpers should be small and deterministic.
- Prefer explicit `Path` usage over string path concatenation.

## Testing Guidelines
- No formal automated test suite exists yet; validate via targeted pipeline runs.
- Add tests under a future `tests/` directory using `test_<module>.py` naming.
- For behavior changes, include at least one reproducible input PDF and verify output JSON shape and key counts (for example, expected question totals).

## Commit & Pull Request Guidelines
- Git history is not available in this workspace, so use Conventional Commit style moving forward: `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`.
- Keep commits scoped to one concern and mention affected stage(s).
- PRs should include: purpose, commands run, sample input used, and before/after notes for output schema or extraction quality.
- Include screenshots only when visual artifacts/composite images are materially affected.

## Security & Configuration Tips
- Do not commit private model endpoints or credentials.
- Override runtime settings through CLI flags (for example, `--ollama-url`, `--model`) instead of hardcoding environment-specific values.
- Avoid unquoted version constraints in ad-hoc shell commands (for example, `pkg>=1.2.3`); shell `>` redirection can create junk files like `=1.2.3`.
