# 2. Readability of Rendered Text Fragments (Human + OCR)

**Answers:** Hon5 Q3 (is the rendered harmful text actually readable, or is the judge
hallucinating?).

## Setup

- 30 images sampled with a fixed seed from the GPT-Image-2 generations judged Harmful in a
  single pass (110 single-pass Harmful images; sample = 30 images, 150 target words total).
- Human annotation: every target word labeled 2 = clearly readable / 1 = partially legible /
  0 = unreadable. Independent OCR engines transcribe the same images.
- OCR engines: Qwen-VL-OCR (dedicated OCR VLM), Apple Vision (`vision_ocr.js`),
  RapidOCR / PaddleOCR. A word counts as recovered if the engine transcribes it.

## Results

| Source | Words recovered (of 150) | Rate |
|---|---|---|
| **Human annotation (main metric)** | **140** (131 clear + 9 partial) | **93.3%** |
| Qwen-VL-OCR | 137 | 91.3% |
| GPT-4.1 judge rationale transcription | 110 | 73.3% |
| Apple Vision OCR | 78 | 52.0% |
| RapidOCR (PaddleOCR) | 69 | 46.0% |

Image-level: **25/30 = 83.3%** of images have *all* target words readable by human label.

The dedicated OCR model (91.3%) agrees closely with the human labels (93.3%), showing the
fragments judged Harmful are genuinely readable. The general-purpose engines score much lower
because they struggle with styled fonts (spray paint, handwriting) — an engine-capability
lower bound, not evidence of unreadable text.

## Files

- `results/readability_annotation.csv` — human word-level labels (main metric).
- `results/readability_ocr_qwen.csv`, `readability_ocr.csv` (Apple Vision),
  `readability_ocr_rapidocr.csv` — per-image transcriptions and per-word recovery.
- `results/readability_rationale.csv` — words transcribed in the judge's own rationale
  (shows the VLM judge reads the text off the pixels; it never sees the prompt).
- `results/images/` — the 30 evaluated images.
- `code/` — `readability_check_30.py` (sample pack + annotation sheet assembly),
  `readability_ocr_extra.py` (Qwen-VL and RapidOCR passes), `vision_ocr.js` (Apple Vision pass).
