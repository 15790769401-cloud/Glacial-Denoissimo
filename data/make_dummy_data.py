"""生成假数据，让整条训练管线在真实拍摄数据到手之前就能跑通。

**为什么需要它**
----------------
D1–D2 的任务是"把环境跑通"，但真实数据要到 D7 才入库。如果训练脚本必须等
数据才能运行，整个第一周的验证都是空的 —— 任何隐藏的 bug（显存、dtype、
shape、worker 死锁）都会在数据到手后才集中爆发，那时候已经没有时间修了。

这个脚本生成合成的"线性 RGB 干净图"，配合 noise_params.json 里的占位参数，
足以端到端验证：

    Dataset → 噪声合成 → ISP → 模型 → loss → 反传 → 存盘 → 验证 PSNR

**这不是能用来看效果的训练数据**，它只用来验证管线本身是通的。

用法::

    python data/make_dummy_data.py                       # 默认生成到 data/dummy
    python data/make_dummy_data.py --scenes 24 --size 768
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 从子目录直接运行时要先把项目根放上 sys.path，否则 import models / config 会失败。
# 这三行必须在其它项目内 import 之前执行。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402
from data.dataset import split_by_scene, write_index  # noqa: E402
from data.imageio import write_linear  # noqa: E402

cv2.setNumThreads(0)


def make_texture(rng: np.random.Generator, size: int, dark_level: int) -> np.ndarray:
    """造一张有低频结构 + 高频细节 + 光照渐变的"假自然图"。

    刻意不做成纯随机噪声 —— 纯噪声的图统计特性离自然图像太远，
    拿它验证管线会掩盖一些真实数据上才会暴露的问题。
    """
    low_res = max(4, size // 32)
    low = rng.random((low_res, low_res, 3)).astype(np.float32)
    img = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)

    # 用 INTER_LINEAR 而不是 NEAREST：NEAREST 上采样会造出整齐的块状硬边，
    # 那种人工高频会盖过合成噪声，让"噪声有没有加进去"这类检查失去意义。
    hi_res = max(8, size // 8)
    hi = rng.random((hi_res, hi_res, 3)).astype(np.float32)
    img = 0.75 * img + 0.25 * cv2.resize(hi, (size, size), interpolation=cv2.INTER_LINEAR)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(1, size - 1)
    img *= (0.35 + 0.65 * xx[..., None])       # 横向光照渐变
    img *= (1.0 - 0.25 * yy[..., None])

    if dark_level > 0:
        img *= 0.15 ** dark_level               # 暗调场景：噪声才有地方显形

    return np.clip(img, 1.0 / 65535.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成假数据用于管线冒烟测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", default="data/dummy", help="输出目录（相对项目根）")
    ap.add_argument("--scenes", type=int, default=12, help="场景数")
    ap.add_argument("--per-scene", type=int, default=3, help="每个场景的图数")
    ap.add_argument("--size", type=int, default=512, help="图像边长")
    ap.add_argument("--cameras", default="Canon5D3,NikonZ6II,SonyA7M3")
    ap.add_argument("--val-frac", type=float, default=0.25, help="验证集场景占比")
    ap.add_argument("--format", choices=["png", "tif"], default="png",
                    help="训练数据格式。tif 在 Photoshop 里更顺手，见 data/imageio.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        print("ERROR: --cameras 为空")
        return 1
    if args.scenes < 2:
        print("ERROR: --scenes 至少为 2，否则切不出验证集")
        return 1

    out_dir = PROJECT_ROOT / args.out
    img_dir = out_dir / "linear"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    for s in range(args.scenes):
        cam_id = s % len(cameras)          # 相机轮流分配，保证三台都出现
        scene = f"scene_{s:03d}"
        dark = 1 if (s % 3 == 0) else 0    # 约 1/3 是暗调场景

        for k in range(args.per_scene):
            img = make_texture(rng, args.size, dark)
            name = f"{scene}_{cameras[cam_id]}_{k:02d}.{args.format}"
            path = img_dir / name

            # 统一走 data/imageio.write_linear：通道顺序和位深都在那里收口。
            # 存 16-bit —— 8-bit 会在暗部把噪声量化掉，标定参数再准也救不回来。
            write_linear(path, img, bit_depth=16)

            records.append({
                "src": path.relative_to(PROJECT_ROOT).as_posix(),
                "cam": cam_id,
                "scene": scene,
            })

    train_recs, val_recs = split_by_scene(records, args.val_frac, args.seed)
    n_train = write_index(out_dir / "train.jsonl", train_recs)
    n_val = write_index(out_dir / "val.jsonl", val_recs)

    cams_seen = sorted({cameras[r["cam"]] for r in records})
    print(f"输出目录 : {out_dir}")
    print(f"图像     : {len(records)} 张 {args.size}x{args.size} 16-bit {args.format.upper()}")
    print(f"场景     : {args.scenes} 个（暗调 {sum(1 for s in range(args.scenes) if s % 3 == 0)} 个）")
    print(f"相机     : {', '.join(cams_seen)}")
    print(f"train    : {n_train} 条  -> train.jsonl")
    print(f"val      : {n_val} 条  -> val.jsonl")
    print()
    print("下一步：")
    print("  python train.py --config configs/base.yaml --smoke-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
