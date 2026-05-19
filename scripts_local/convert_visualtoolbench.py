"""
将 VisualToolBench (parquet, ScaleAI/VisualToolBench) 转换成 XSkill 期望的格式：

输出：
  benchmark/VisualToolBench/
    images/<doc_id>/img_<i>.<ext>          # 图片落地
    val_full.json                           # 完整 1204 条
    val_single.json                         # 仅单轮 (~603 条，简单)
    val_multi.json                          # 仅多轮 (~601 条)
    val_50.json                             # 前 50 条用于 sanity check

XSkill 单条样本格式：
  {
    "doc_id":   "<id>",
    "problem":  "<image>\n<turn1 prompt>",
    "images":   ["VisualToolBench/images/<doc_id>/img_0.png", ...],
    "solution": "<turn1 golden answer>",
    "data_source": "<prompt_category>"
  }

注意：
- VisualToolBench 是多轮的，但 XSkill 评测代码只看第 0 轮。所以这里保守做法 =
  把每个样本展开成 "用第 1 轮 prompt + golden 作为 problem/solution"，多轮信息
  存到 raw_extra 字段（供后续 rubric 评分扩展）。
- <image> 占位符按图片数量插入到 prompt 开头，匹配 XSkill 的 prompt 拼接逻辑。
"""

import os
import json
import argparse
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO


def _save_image(img_obj, target_path: Path) -> bool:
    """img_obj 可能是 dict({'bytes': ..., 'path': ...}) 或 PIL.Image，都处理。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(img_obj, dict):
            data = img_obj.get('bytes')
            if data:
                img = Image.open(BytesIO(data))
            elif img_obj.get('path'):
                img = Image.open(img_obj['path'])
            else:
                return False
        elif hasattr(img_obj, 'save'):
            img = img_obj
        else:
            return False
        if img.mode in ('P', 'RGBA') and target_path.suffix.lower() in ('.jpg', '.jpeg'):
            img = img.convert('RGB')
        img.save(target_path)
        return True
    except Exception as e:
        print(f"  [WARN] save image fail: {e}")
        return False


def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        # take first non-empty
        for it in x:
            if it:
                return str(it)
        return ""
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', default='/data2/chenxuwu/zihaowan_workplace/visualtoolbench',
                    help='downloaded HF snapshot dir')
    ap.add_argument('--out-dir', default='/data2/chenxuwu/zihaowan_workplace/XSkill/benchmark/VisualToolBench',
                    help='where to write XSkill-formatted data')
    ap.add_argument('--max-samples', type=int, default=-1,
                    help='limit total processed samples (-1 = all)')
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    img_root = out / 'images'
    out.mkdir(parents=True, exist_ok=True)
    img_root.mkdir(parents=True, exist_ok=True)

    # 找所有 parquet
    parquets = sorted(src.glob('**/*.parquet'))
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {src}")
    print(f"Found {len(parquets)} parquet file(s):")
    for p in parquets:
        print(f"  - {p.relative_to(src)}  ({p.stat().st_size/1e6:.1f} MB)")

    all_samples = []
    total_rows = 0
    for pq_path in parquets:
        print(f"\nReading {pq_path.name}...")
        table = pq.read_table(pq_path)
        df_rows = table.to_pylist()
        print(f"  rows: {len(df_rows)}")
        for row in df_rows:
            total_rows += 1
            if args.max_samples > 0 and len(all_samples) >= args.max_samples:
                break

            doc_id = _safe_str(row.get('id'))
            if not doc_id:
                continue
            turncase = _safe_str(row.get('turncase'))
            num_turns = int(row.get('num_turns') or 1)
            prompt_category = _safe_str(row.get('prompt_category'))
            eval_focus = _safe_str(row.get('eval_focus'))
            turn_prompts = row.get('turn_prompts') or []
            turn_golden = row.get('turn_golden_answers') or []
            turn_traj = row.get('turn_tool_trajectories') or []
            rubrics_by_turn = row.get('rubrics_by_turn') or []
            images_flat = row.get('images') or []
            images_by_turn = row.get('images_by_turn') or []

            # 落图（命名为 img_0, img_1, ...）
            saved_paths = []
            for idx, im in enumerate(images_flat):
                # 决定扩展名：默认 png 安全
                ext = '.png'
                if isinstance(im, dict):
                    p = im.get('path') or ''
                    if p:
                        e = os.path.splitext(p)[1].lower()
                        if e in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'):
                            ext = e
                target = img_root / doc_id / f"img_{idx}{ext}"
                if not target.exists():
                    if not _save_image(im, target):
                        continue
                # 相对于 IMAGE_DIR=benchmark 的路径
                rel = f"VisualToolBench/images/{doc_id}/img_{idx}{ext}"
                saved_paths.append(rel)

            # 第一轮 prompt + golden（XSkill 评测只用第 0 轮）
            first_prompt = turn_prompts[0] if turn_prompts else ""
            first_golden = turn_golden[0] if turn_golden else ""

            # 在 prompt 头补 <image> 占位（按本轮的图片数）
            n_imgs_first_turn = (
                len(images_by_turn[0]) if images_by_turn and isinstance(images_by_turn[0], list)
                else len(saved_paths)
            )
            placeholders = "\n".join(["<image>"] * n_imgs_first_turn)
            full_prompt = (placeholders + "\n" + first_prompt).strip() if placeholders else first_prompt

            # 这一轮要用的图片路径子集（和 problem 里 <image> 数量对齐）
            this_turn_paths = saved_paths[:n_imgs_first_turn] if saved_paths else []

            sample = {
                "doc_id": doc_id,
                "problem": full_prompt,
                "images": this_turn_paths,
                "solution": first_golden,
                "data_source": prompt_category or eval_focus or "visualtoolbench",
                # 额外字段，留给后续 rubric 评分使用
                "raw_extra": {
                    "turncase": turncase,
                    "num_turns": num_turns,
                    "prompt_category": prompt_category,
                    "eval_focus": eval_focus,
                    "all_turn_prompts": turn_prompts,
                    "all_turn_goldens": turn_golden,
                    "all_turn_tool_trajectories": turn_traj,
                    "all_rubrics_by_turn": rubrics_by_turn,
                    "all_image_paths": saved_paths,
                },
            }
            all_samples.append(sample)
            if total_rows % 100 == 0:
                print(f"  processed {total_rows} rows, kept {len(all_samples)}...")

    # 三个分组
    val_full = all_samples
    val_single = [s for s in all_samples if s['raw_extra']['turncase'] == 'single-turn']
    val_multi = [s for s in all_samples if s['raw_extra']['turncase'] == 'multi-turn']
    val_50 = all_samples[:50]

    for fname, lst in [
        ('val_full.json', val_full),
        ('val_single.json', val_single),
        ('val_multi.json', val_multi),
        ('val_50.json', val_50),
    ]:
        with open(out / fname, 'w', encoding='utf-8') as f:
            json.dump(lst, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {out/fname}: {len(lst)} samples")

    print(f"\nDone. total_rows seen={total_rows}, total_kept={len(all_samples)}")
    print(f"  single-turn: {len(val_single)}")
    print(f"  multi-turn:  {len(val_multi)}")
    print(f"Image dir: {img_root}")


if __name__ == '__main__':
    main()
