# 2. Readability of Rendered Text Fragments (Human + OCR)

**Answers:** Hon5 Q3 (is the rendered harmful text actually readable, or is the judge
hallucinating?).


## Results

| Source | Words recovered (of 150) | Rate |
|---|---|---|
| **Human annotation (main metric)** | **140** (131 clear + 9 partial) | **93.3%** |
| Qwen-VL-OCR | 137 | 91.3% |
| GPT-4.1 judge rationale transcription | 110 | 73.3% |
| Apple Vision OCR | 78 | 52.0% |
| RapidOCR (PaddleOCR) | 69 | 46.0% |


The dedicated OCR model (91.3%) agrees closely with the human labels (93.3%), showing the
fragments judged Harmful are genuinely readable. The general-purpose engines score much lower
because they struggle with styled fonts (spray paint, handwriting) — an engine-capability
lower bound, not evidence of unreadable text.

