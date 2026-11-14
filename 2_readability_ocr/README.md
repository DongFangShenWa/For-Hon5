# 2. Readability of Rendered Text Fragments (Human + OCR)

**Answers:** Hon5 Q3 (is the rendered harmful text actually readable, or is the judge hallucinating?).

## Results

| Source | Word recovery rate | Notes |
|---|---:|---|
| **Human annotation (main metric)** | **93.3%** | Most words were clearly readable; a small portion was partially legible. |
| Qwen-VL-OCR | 91.3% | Specialized OCR model; closely agrees with human annotation |
| GPT-4.1 judge rationale transcription | 73.3% | Transcription by the judge model |
| Apple Vision OCR | 52.0% | General-purpose OCR |
| RapidOCR (PaddleOCR) | 46.0% | General-purpose OCR |

The dedicated OCR model agrees closely with the human labels, supporting that the fragments judged Harmful are genuinely readable. The general-purpose engines score lower, likely because they struggle with styled fonts such as spray paint and handwriting; this reflects engine limitations and does not by itself show that the text is unreadable.
