# codex-ocr-prj

OCR pipeline for mathematics question papers:

1. Stage 1: PDF -> Markdown + extracted images (`stage1_marker.py`)
2. Stage 2: Markdown -> structured JSON (`stage2_ollama.py`)
3. Stage 3: JSON -> answered JSON (`stage3_ollama.py`)

`main.py` runs the full pipeline (or up to a selected stage).

## Prerequisites

- Python 3.10+ (recommended)
- A running Ollama server
- Ollama model available locally: `llama3.2:3b`

Example:

```bash
ollama pull llama3.2:3b
ollama serve
```

## Installation

### Option A (recommended)

```bash
make install
```

This creates `.venv`, upgrades `pip`, and installs dependencies from `requirements.txt`.

### Option B (manual)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Full Pipeline

All examples below write outputs to `output-final/`.

### Input type: PDF

```bash
.venv/bin/python main.py \
  --pdf-path input-pdf/65-2-2_Mathematics.pdf \
  --output-dir output-final
```

### Input type: one or more images

Single image:

```bash
.venv/bin/python main.py \
  --image-path images/page1.png \
  --output-dir output-final
```

Multiple images (they are merged into one generated PDF first):

```bash
.venv/bin/python main.py \
  --image-path images/page1.png images/page2.png images/page3.png \
  --output-dir output-final
```

## Useful Runtime Options (`main.py`)

- `--stage 1|2|3`: run only up to that stage (default `3`)
- `--series <value>`: override inferred paper series
- `--ollama-url <url>`: default is `http://192.168.2.156:11434`
- `--stage2-model <name>`: default `llama3.2:3b`
- `--stage3-model <name>`: default `llama3.2:3b`
- `--model <name>`: deprecated shortcut to set both Stage 2 and Stage 3 model
- `--num-ctx <int>`: context window (default `32768`)
- `--num-predict <int>`: max generated tokens (default `8192` in pipeline)
- `--wipe`: clear output directory before run
- `--cpu-only`: run marker on CPU (slower, safer on constrained GPUs)
- `--disable-low-vram-mode`: disables conservative marker settings
- `--debug`: print Stage 3 request/response debug logs

## Run Individual Stages

### Stage 1 only: PDF -> Markdown + images

```bash
.venv/bin/python stage1_marker.py \
  input-pdf/65-2-2_Mathematics.pdf \
  output \
  --series 65-2-2
```

Optional CPU mode:

```bash
.venv/bin/python stage1_marker.py \
  input-pdf/65-2-2_Mathematics.pdf \
  output \
  --series 65-2-2 \
  --cpu-only
```

Optional fast-path controls:

- `--disable-fast-text-path`: force marker-based extraction (disable `pdftotext` shortcut)
- `--extract-images-fast-path`: when fast text path is used, also extract embedded PDF images (slower)
- `--force-fast-text-path`: force `pdftotext` output even if coverage checks detect possible partial extraction

### Stage 2 only: Markdown -> structured JSON

```bash
.venv/bin/python stage2_ollama.py \
  output/65-2-2.md \
  output/65-2-2.json \
  --series 65-2-2 \
  --ollama-url http://127.0.0.1:11434 \
  --model llama3.2:3b
```

Optional refinement pass:

```bash
.venv/bin/python stage2_ollama.py \
  output/65-2-2.md \
  output/65-2-2.json \
  --series 65-2-2 \
  --use-ollama-refine
```

### Stage 3 only: structured JSON -> answered JSON

```bash
.venv/bin/python stage3_ollama.py \
  output/65-2-2.json \
  output/65-2-2_with_answer.json \
  --ollama-url http://127.0.0.1:11434 \
  --model llama3.2:3b
```

## Output Files

For series `65-2-2` and output directory `output-final/`:

- `output-final/65-2-2.md`
- `output-final/65-2-2_images/`
- `output-final/65-2-2.json`
- `output-final/65-2-2_with_answer.json`

## Cleanup

If shell redirection mistakes created files like `=1.2.3`, clean them with:

```bash
make clean-leaks
```
