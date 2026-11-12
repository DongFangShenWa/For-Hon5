# DACA × GPT-Image-2 前 10 条基线结果

- 数据：sampled_200_hate.csv（前 50 句 hate）
- backbone：gpt-4.1（DACA 模板与本体图原封未动）
- 生图：gpt-image-2-2026-04-21（quality=medium）
- 评审：gpt-4.1 论文同款 rubric（0–10 分）

## 结果

| 指标 | 值 |
| --- | --- |
| 变换失败 | 5/50 |
| 输入被拦截（safety_rejected） | 4/50 |
| 成功生成图片 | 41/50 |
| ASR（Harmful/总数） | 7/50 = 14.0% |
| 平均有害分（成功图） | 0.93/10 |

逐条明细见 items.csv。