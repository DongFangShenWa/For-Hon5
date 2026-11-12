# -*- coding: utf-8 -*-
"""
CollageAttack x Stable Diffusion XL base 1.0 (本地权重, 服务器 GPU 推理)

与 attack_model_sd35_large.py / attack_model_example.py 同构:
  - 输入: coj_commands.csv (prompt_generation.py 的产物)
  - 提示词拼接: command1/command2/command3 以 ". " 连接(空值跳过)
  - 输出: attack_results/sdxl-base-1.0/combined_hate_XX.png
  - 失败 index 记录: attack_results/sdxl-base-1.0/failed_indices.txt
后续用 judge_image_score.py(MODEL_NAME=sdxl-base-1.0)评测。

SDXL 与 SD3.5 的关键差异:
  - 无 T5 编码器, 双 CLIP 上限 77 token: ~362 token 的攻击提示词只送入前 77 token,
    command2 的文字书写指令基本被截断——SDXL 架构限制, 属于实验本身的一部分;
  - 官方默认 50 步 / CFG 5.0;
  - fp16 约 7GB, 24GB 卡无需 cpu_offload。

示例:
  python attack_model_sdxl.py --limit 10          # 冒烟
  python attack_model_sdxl.py                     # 全量
"""

import argparse
import os
import time

import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline

# 77-token 截断是预期行为且无法避免, 关掉 transformers 的重复警告刷屏(错误仍会正常抛出)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


# ================= 配置区(命令行参数可覆盖) =================
def parse_args():
    parser = argparse.ArgumentParser(description="CollageAttack on SDXL-base-1.0 (local weights)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/hub/huggingface/models/stabilityai/stable-diffusion-xl-base-1.0",
        help="本地模型路径或 HF repo id",
    )
    parser.add_argument("--input_csv", type=str, default="coj_commands.csv")
    parser.add_argument("--prompt_column", type=str, default=None,
                        help="单列 prompt 模式(baseline 实验用): 直接取该列作为提示词, "
                             "如 --input_csv baseline_prompts/pgj_prompts.csv --prompt_column prompt; "
                             "不设则用 command1/2/3 三段拼接(CollageAttack 模式)")
    parser.add_argument("--results_dir", type=str, default="attack_results/sdxl-base-1.0")
    parser.add_argument("--failed_indices_file", type=str, default=None,
                        help="默认取 results_dir/failed_indices.txt")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="负向提示词(默认为空, 与 API 版攻击脚本保持一致)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50, help="推理步数, SDXL 官方默认 50")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG, SDXL 官方默认 5.0")
    parser.add_argument("--seed", type=int, default=42, help="基础种子, 实际种子 = seed + image_index")
    parser.add_argument("--start_index", type=int, default=1,
                        help="从第几条开始(1-based), 用于手工分段")
    parser.add_argument("--limit", type=int, default=0,
                        help="本次最多处理多少条(0=不限), 配合 start_index 做冒烟/分段")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--retry_sleep", type=int, default=3)
    parser.add_argument("--cpu_offload", action="store_true",
                        help="SDXL fp16 约 7GB, 24GB 卡一般用不上; 显存异常紧张时才开")
    parser.add_argument("--overwrite", action="store_true",
                        help="忽略已存在图片, 强制重新生成")
    return parser.parse_args()


# ================= 工具函数 =================
def append_failed_index(failed_file, image_index, reason=""):
    os.makedirs(os.path.dirname(failed_file), exist_ok=True)
    with open(failed_file, "a") as f:
        f.write(f"{image_index}\t{reason}\n")


def is_cuda_oom_error(e) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg or "alloc" in msg and "memory" in msg


def _clean_cell(value) -> str:
    """与 prompt_generation.py 的 clean_cell 一致: None/NaN/'None' 统一为空"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower().rstrip(".") == "none":
        return ""
    return text


def build_combined_prompt(row, prompt_column=None) -> str:
    """CollageAttack 模式: command1/2/3 以 '. ' 连接(空值跳过, 168/200 条 command3 为空)。
    baseline 模式(prompt_column 非空): 直接取该列(空值/NaN 报错而非静默跳过)。"""
    if prompt_column:
        text = _clean_cell(row[prompt_column])
        if not text:
            raise ValueError(f"prompt_column={prompt_column} 出现空值, index 行数据异常")
        return text
    parts = [_clean_cell(row[c]) for c in ("command1", "command2", "command3")]
    return ". ".join([p for p in parts if p])


def load_pipeline(args, device):
    print(f"加载模型: {args.model_path}")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # 共享盘 /hub 是 NFS, 偶发属性缓存抖动会漏掉组件目录(如 unet), 加载失败时退避重试
    pipe, last_err = None, None
    for attempt in range(1, 6):
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.model_path,
                torch_dtype=dtype,
                local_files_only=os.path.isdir(args.model_path),
            )
            break
        except ValueError as e:
            last_err = e
            print(f"  -> 加载失败 (尝试 {attempt}/5): {str(e)[:150]}")
            print("     (NFS 抖动概率大, 30s 后重试)")
            time.sleep(30)
    if pipe is None:
        raise SystemExit(f"模型加载连续 5 次失败, 最后错误: {last_err}")
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        try:
            pipe = pipe.to(device)
        except torch.OutOfMemoryError:
            raise SystemExit(
                "\n!! CUDA OOM: SDXL fp16 约 7GB, 一般 24GB 卡不会 OOM;"
                "   若卡被其他进程占用, 可加 --cpu_offload 重跑。\n"
            )
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ================= 主逻辑 =================
def run_combined_attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("!! 未检测到 CUDA, SDXL 在 CPU 上极慢, 建议使用 GPU 服务器")

    os.makedirs(args.results_dir, exist_ok=True)
    failed_file = args.failed_indices_file or os.path.join(args.results_dir, "failed_indices.txt")

    df = pd.read_csv(args.input_csv)
    required = [args.prompt_column] if args.prompt_column else ["command1", "command2", "command3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入 CSV 缺少列 {missing}, 实际列: {list(df.columns)}")

    total_count = len(df)
    end_index = total_count if args.limit <= 0 else min(args.start_index - 1 + args.limit, total_count)

    pipe = load_pipeline(args, device)

    success_count = 0
    skipped_count = 0
    failed_count = 0
    t_start = time.time()

    print(f"\n开始【SDXL-base-1.0 本地推理】实验, 共 {total_count} 条, "
          f"本次处理 [{args.start_index}, {end_index}]...")
    print("(注意: SDXL 双 CLIP 编码器上限 77 token, 长攻击提示词会被截断, 属架构限制)\n")

    for idx in range(args.start_index - 1, end_index):
        image_index = idx + 1
        row = df.iloc[idx]
        combined_prompt = build_combined_prompt(row, args.prompt_column)
        save_path = os.path.join(args.results_dir, f"combined_hate_{image_index:02d}.png")

        print(f"[{image_index:02d}/{total_count}] 提示词: {combined_prompt[:80]}...")

        # 断点续跑: 已生成过的直接跳过
        if os.path.exists(save_path) and not args.overwrite:
            print("  -> 已存在, 跳过 ⏭")
            skipped_count += 1
            continue

        success = False
        for attempt in range(1, args.max_attempts + 1):
            try:
                tag = f"(第{attempt}次)" if attempt > 1 else ""
                print(f"  -> 生成中{tag} ...")

                generator = torch.Generator(device=device.type).manual_seed(args.seed + image_index)
                image = pipe(
                    prompt=combined_prompt,
                    negative_prompt=args.negative_prompt or None,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]

                image.save(save_path)
                print(f"  -> 生成成功 ✅ -> {save_path}")
                success_count += 1
                success = True
                break

            except Exception as e:
                if is_cuda_oom_error(e):
                    print(f"  -> CUDA OOM, 清理显存后重试 ({attempt}/{args.max_attempts})")
                    torch.cuda.empty_cache()
                    time.sleep(args.retry_sleep)
                    continue
                print(f"  -> 报错 (尝试 {attempt}/{args.max_attempts}): {type(e).__name__}: {e}")
                if attempt < args.max_attempts:
                    print(f"  -> {args.retry_sleep}s 后重试...")
                    time.sleep(args.retry_sleep)

        if not success:
            print(f"  -> 已尝试 {args.max_attempts} 次均失败, 记录 index")
            failed_count += 1
            append_failed_index(failed_file, image_index, "generation_error")
            torch.cuda.empty_cache()

        print()

    elapsed = time.time() - t_start
    print("=" * 46)
    print("SDXL-base-1.0 本地推理 实验结果统计报告")
    print("=" * 46)
    print(f"总数据条数: {total_count}")
    print(f"本次范围: [{args.start_index}, {end_index}]")
    print(f"生成成功数: {success_count}")
    print(f"跳过数(已存在): {skipped_count}")
    print(f"失败数: {failed_count}")
    print(f"耗时: {elapsed / 60:.1f} min, 平均 {elapsed / max(success_count, 1):.1f} s/张")
    if failed_count > 0:
        print(f"失败 index 已保存至: {failed_file}")
    print("=" * 46)


if __name__ == "__main__":
    run_combined_attack(parse_args())
