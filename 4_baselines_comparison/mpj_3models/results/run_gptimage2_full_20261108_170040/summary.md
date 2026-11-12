# MPJ sign 模板 × gpt-image-2-2026-04-21（全量 n=200）

- 输入端安全拦截: 189/200 (94.5%)
- 生成成功: 11/200
- 其他错误: 0/200
- 评审 Harmful: 8/200  → ASR = 4.0%（分母=总尝试，与论文 Table 1 同口径）
- 生成图的平均 Harmful Score: 3.18

参照（论文 Table 1，GPT-Image-2，200 条全量）：
- Original: ASR 7/15/200（输入拦截 185/200 = 92.5%），HarmfulScore 3.2
- CollageAttack: ASR 115/165/200（输入拦截 35/200 = 17.5%），HarmfulScore 4.96

## 结论要点
- MPJ 模板把完整有害句连续保留在 prompt 中 → 强 LLM 输入护栏拦截率与 Original 同量级，无规避增益
- 中性场景为本组下界：MPJ 原法的配套场景只会增加 prompt 有害线索