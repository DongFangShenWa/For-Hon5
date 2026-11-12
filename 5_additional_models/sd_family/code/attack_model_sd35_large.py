# -*- coding: utf-8 -*-
"""
CollageAttack x Stable Diffusion 3.5 Large (本地权重, 服务器 GPU 推理)

用法与 attack_model_example.py 完全同构:
  - 输入: coj_commands.csv (prompt_generation.py 的产物)
  - 提示词拼接: f"{command1}. {command2}. {command3}."
  - 输出: attack_results/sd3.5-large/combined_hate_XX.png
  - 失败 index 记录: attack_results/sd3.5-large/failed_indices.txt
后续可直接沿用 judge_image_score.py / judge_image_text_comparison.py /
clip_similarity_eval_oneclick_with_cache.py 评测。

示例:
  # 全量 200 条
  python attack_model_sd35_large.py \
      --model_path /hub/huggingface/models/stabilityai/stable-diffusion-3.5-large

  # 冒烟测试前 10 条 + 显存不够时开 CPU offload
  python attack_model_sd35_large.py --limit 10 --cpu_offload

  # 断点续跑: 已存在的 combined_hate_XX.png 会自动跳过
"""

import argparse
import os
import time

import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline


# ================= 配置区(命令行参数可覆盖) =================
def parse_args():
    parser = argparse.ArgumentParser(description="CollageAttack on SD 3.5 Large (local weights)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/hub/huggingface/models/stabilityai/stable-diffusion-3.5-large",
        help="本地模型路径或 HF repo id",
    )
    parser.add_argument("--input_csv", type=str, default="coj_commands.csv")
    parser.add_argument("--prompt_column", type=str, default=None,
                        help="单列 prompt 模式(baseline 实验用): 直接取该列作为提示词, "
                             "如 --input_csv baseline_prompts/pgj_prompts.csv --prompt_column prompt; "
                             "不设则用 command1/2/3 三段拼接(CollageAttack 模式)")
    parser.add_argument("--results_dir", type=str, default="attack_results/sd3.5-large")
    parser.add_argument("--failed_indices_file", type=str, default=None,
                        help="默认取 results_dir/failed_indices.txt")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="负向提示词(默认为空, 与 API 版攻击脚本保持一致)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28, help="推理步数, SD3.5 官方默认 28")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="CFG, SD3.5 官方默认 7.0")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="T5 编码器最大 token 数(diffusers 默认 256, 攻击提示词约 360 token, "
                             "command2 的文字指令会被截断, 调到 512 保证完整送入; "
                             "CLIP 的 77 上限是架构硬限制, 其截断警告属正常)")
    parser.add_argument("--seed", type=int, default=42, help="基础种子, 实际种子 = seed + image_index")
    parser.add_argument("--start_index", type=int, default=1,
                        help="从第几条开始(1-based), 用于手工分段")
    parser.add_argument("--limit", type=int, default=0,
                        help="本次最多处理多少条(0=不限), 配合 start_index 做冒烟/分段")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--retry_sleep", type=int, default=3)
    parser.add_argument("--cpu_offload", action="store_true",
                        help="显存不足时开启 enable_model_cpu_offload()")
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
    # 共享盘 /hub 是 NFS, 偶发属性缓存抖动会漏掉组件目录, 加载失败时退避重试
    pipe, last_err = None, None
    for attempt in range(1, 6):
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
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
                "\n!! CUDA OOM: 24GB 单卡装不下 SD3.5-Large 全量 fp16 管线(约 27GB)。\n"
                "   加 --cpu_offload 重跑即可(组件按需上卡, 推理时 transformer 独占 GPU)。\n"
            )
    # 关闭 NSFW 检查的水印即可, SD3.5 默认无 safety_checker, 无需额外处理
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ================= 主逻辑 =================
def run_combined_attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("!! 未检测到 CUDA, SD3.5-Large 在 CPU 上极慢, 建议使用 GPU 服务器")

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

    # 生成器要在设备上; 每条 prompt 用 seed+index 固定随机种子, 保证可复现
    success_count = 0
    skipped_count = 0
    failed_count = 0
    t_start = time.time()

    print(f"\n开始【SD3.5-Large 本地推理】实验, 共 {total_count} 条, "
          f"本次处理 [{args.start_index}, {end_index}]...\n")

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
                    max_sequence_length=args.max_seq_len,
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
    print("SD3.5-Large 本地推理 实验结果统计报告")
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
