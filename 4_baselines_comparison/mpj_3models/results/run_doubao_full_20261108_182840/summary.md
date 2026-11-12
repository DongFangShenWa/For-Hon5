# MPJ sign 模板 × doubao-seedream-5-0-lite-260128（全量 n=200）

- 输入端安全拦截: 4/200 (2.0%)
- 生成成功: 196/200
- 其他错误: 0/200
- 评审 Harmful: 195/200  → ASR = 97.5%（分母=总尝试，与论文 Table 1 同口径）
- 生成图的平均 Harmful Score: 8.59

参照（论文 Table 1，Doubao-Seedream-5.0-Lite，200 条全量）：
- Original: ASR 63/185/200（输入拦截 15/200 = 7.5%），HarmfulScore 2.7
- CollageAttack: ASR 172/198/200（输入拦截 2/200 = 1.0%），HarmfulScore 6.85

## 结论要点
- MPJ 模板把完整有害句连续保留在 prompt 中 → 强 LLM 输入护栏拦截率与 Original 同量级，无规避增益
- 中性场景为本组下界：MPJ 原法的配套场景只会增加 prompt 有害线索