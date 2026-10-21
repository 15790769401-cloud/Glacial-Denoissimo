"""把文件索引打包成 LMDB，消除训练时的解码瓶颈。

**我们已经量过了，所以默认就用它**（2026-10 实测）：

============================  ============  ==========
后端                          每优化步       GPU
============================  ============  ==========
直接读 24MP 16-bit TIFF       4.13 s        很低，GPU 一直在等
LMDB（预切好的 patch）        **0.79 s**    82%
============================  ============  ==========

一张 24MP 的 TIFF 读一次 0.9 秒，而一个优化步要吃 16 个样本
（batch 4 × accum 4）—— 光数据加载就 3.2 秒，比模型本身还慢。
LMDB 每次只取一个 128×128 的 patch（98KB），整体快 5 倍。
**这不是"优化"，是能不能在截止日前跑完的差别**（300k 步：66h vs 343h）。

代价：patch 是**预先裁好**的，一张原图裁多少张要提前定死
（`--patches-per-image`，默认 32，实测用 64）。噪声仍然是每个 epoch
在线合成，所以噪声多样性不受影响；空间多样性靠 64 个不同裁剪位置 +
翻转旋转增强来保证。

⚠️ **改了 `--patches-per-image`、或者 `noise_params.json` 里的相机变了，
都要重跑本脚本。** 索引里的相机下标是按当时的标定文件算的，
`dataset.py` 启动时会核对，对不上直接报错（不会静默跑错）。

**真实配对组（phase 2 微调用）**：输入索引里带 `noisy_src` 的记录，
这里会把 GT 和拍出来的噪声图**用同一组 (y,x)** 各切一份存进库，
记录里带 `noisy_src`（此时是 **LMDB key**，不是文件路径）/ `iso`。
`dataset.py` 看到这个字段就直接读真实噪声，不再在线合成。

体积估算：128x128x3x2B = 98KB/patch。

用法::

    python data/build_lmdb.py --index data/real/train.jsonl \\
        --out data/real/train.lmdb --patches-per-image 64
    python data/build_lmdb.py --index data/real/val.jsonl \\
        --out data/real/val.lmdb  --patches-per-image 32

输出索引是 `<out>.jsonl`（例如 `train.lmdb.jsonl`），config 里填那个。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402
from data.dataset import load_index, write_index  # noqa: E402
from data.imageio import read_linear  # noqa: E402

cv2.setNumThreads(0)

PATCH_BYTES = lambda p: p * p * 3 * 2  # 128x128x3 uint16


def _resolve(src: str) -> Path:
    p = Path(src)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PNG 索引 → LMDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--index", required=True, help="输入索引 jsonl")
    ap.add_argument("--out", required=True, help="输出 .lmdb 目录")
    ap.add_argument("--out-index", default=None,
                    help="输出索引 jsonl（默认在 --out 旁边同名 .jsonl）")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--patches-per-image", type=int, default=32)
    ap.add_argument("--map-size-gb", type=float, default=None,
                    help="LMDB map size，默认按记录数自动估算并留 1 倍余量"
                         "（写时复制会让峰值逼近两倍，余量小了会 MDB_MAP_FULL）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import lmdb

    records = load_index(_resolve(args.index))
    out_lmdb = _resolve(args.out)

    if args.out_index:
        out_index = _resolve(args.out_index)
    else:
        # ⚠️ 默认输出名**不能**用 out_lmdb.with_suffix(".jsonl")。
        #    那是 `train.lmdb` → `train.jsonl`，正好就是输入索引本身 ——
        #    等于把源索引原地覆盖掉，之后想换 patches-per-image 重建就没有输入了，
        #    而且 png 后端也再回不去。改成在后面接一段，变成 train.lmdb.jsonl。
        out_index = _resolve(str(out_lmdb) + ".jsonl")

    if out_index.resolve() == _resolve(args.index).resolve():
        print(f"ERROR: --out-index 和 --index 是同一个文件（{out_index}）")
        print("       继续下去会把输入索引覆盖掉，已中止。")
        return 1

    # ★ 配对组（带 noisy_src 的记录）在库里存【两份】patch —— GT 一份、
    #   真实噪声图一份 —— 体积要一起算，否则 map_size 会估小。
    n_pair = sum(1 for r in records if r.get("noisy_src"))
    est = (len(records) + n_pair) * args.patches_per_image * PATCH_BYTES(args.patch)
    # 留 2 倍余量，不是 1.5 倍：写的时候 LMDB 用写时复制，一整个事务里
    # 「旧页 + 新页」会同时存在，峰值真的会逼近两倍。1.5 倍实测会炸。
    map_size = int(args.map_size_gb * 1024**3) if args.map_size_gb else int(est * 2.0)
    map_size = max(map_size, 64 * 1024**2)

    print(f"输入     : {len(records)} 条" +
          (f"（其中真实配对 {n_pair} 条，各存 GT + 噪声两份 patch）" if n_pair else ""))
    print(f"输出     : {out_lmdb}")
    print(f"预计体积 : {est / 1024**3:.2f} GB（map_size 设为 {map_size / 1024**3:.2f} GB）")

    rng = np.random.default_rng(args.seed)
    new_records: list[dict] = []
    skipped = 0

    # ★ 必须从空库开始写。
    #   本脚本是"全量重建"——每次都会重新生成全部 patch 和索引，往旧库里
    #   覆盖写毫无意义，而且会直接把 map_size 撑爆：
    #       lmdb.MapFullError: mdb_put: MDB_MAP_FULL: Environment mapsize limit reached
    #   实测：0.91GB 的旧库 + 一次 314MB 的写事务 ≈ 1.23GB 峰值 > 1.27GB 上限。
    #   报错信息看着像"库太大"，其实是"没清空"。
    if out_lmdb.exists() and any(out_lmdb.iterdir()):
        print(f"检测到已存在的 LMDB，全量重建会先清空它：{out_lmdb}")
        shutil.rmtree(out_lmdb)

    out_lmdb.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_lmdb), map_size=map_size)
    try:
        txn = env.begin(write=True)
        done = 0
        for i, rec in enumerate(records):
            path = _resolve(str(rec["src"]))
            try:
                # ★ 必须走 read_linear：它返回的是【RGB 顺序】。
                #   直接用 cv2.imread 拿到的是 BGR，存进 LMDB 后
                #   dataset 那边按 RGB 解释 —— 走 LMDB 后端时 R/B 会互换，
                #   而且图能正常打开，只是颜色偏，极难发现。
                linear = read_linear(path)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  [跳过] {path.name}: {exc}")
                skipped += 1
                continue

            img = (np.clip(linear, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
            h, w = img.shape[:2]
            if h < args.patch or w < args.patch:
                print(f"  [跳过] {path.name} 只有 {h}x{w}，小于 patch {args.patch}")
                skipped += 1
                continue

            # ★ 配对组：把**拍出来的真实噪声图**也一起切进 LMDB。
            #   关键点：GT 和噪声必须用**同一组 (y,x)** 裁下来，否则两张图
            #   对不上 —— 模型会拿"A 处的干净图"去学"B 处的噪声"，
            #   loss 照样降，图却是错的。dataset 那边对 patch 大小的输入
            #   返回 (0,0) 裁剪，所以这里对齐了，那边就自动保持。
            noisy_img = None
            if rec.get("noisy_src"):
                try:
                    nlin = read_linear(_resolve(str(rec["noisy_src"])))
                except (FileNotFoundError, ValueError) as exc:
                    print(f"  [跳过] {path.name} 的配对噪声图读不了: {exc}")
                    skipped += 1
                    continue
                if nlin.shape[:2] != (h, w):
                    print(f"  [跳过] {path.name}: GT {h}x{w} 与配对噪声图 "
                          f"{nlin.shape[:2]} 尺寸不一致")
                    skipped += 1
                    continue
                noisy_img = (np.clip(nlin, 0.0, 1.0) * 65535.0).round().astype(np.uint16)

            for j in range(args.patches_per_image):
                y = int(rng.integers(0, h - args.patch + 1))
                x = int(rng.integers(0, w - args.patch + 1))
                patch = np.ascontiguousarray(img[y:y + args.patch, x:x + args.patch])
                key = f"{i:07d}_{j:03d}"
                txn.put(key.encode("utf-8"), patch.tobytes())
                entry = {
                    "src": key,
                    "cam": int(rec["cam"]),
                    "scene": str(rec["scene"]),
                    "h": args.patch,
                    "w": args.patch,
                    # 相机名/category 透传下去：dataset 启动时会用它反查下标对不对，
                    # 报告也要按相机分组统计。
                    **{k: rec[k] for k in ("camera", "category", "wb") if k in rec},
                }
                if noisy_img is not None:
                    # 同一坐标切噪声图，另存一个 key（后缀 _n，不会和 GT 撞）
                    nkey = f"{i:07d}_{j:03d}_n"
                    npatch = np.ascontiguousarray(
                        noisy_img[y:y + args.patch, x:x + args.patch]
                    )
                    txn.put(nkey.encode("utf-8"), npatch.tobytes())
                    # ⚠️ 这里的 noisy_src 是 **LMDB key**，不是文件路径。
                    #    dataset._load_source() 在 lmdb 后端下按 key 取，
                    #    所以语义是对的；但**不能**再拿这个字段当路径用。
                    entry["noisy_src"] = nkey
                    entry["noisy_h"] = args.patch
                    entry["noisy_w"] = args.patch
                    if "iso" in rec:
                        entry["iso"] = int(rec["iso"])   # 真实噪声图的 ISO，作条件输入
                new_records.append(entry)

            done += 1
            if done % 50 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"  已处理 {done}/{len(records)} 张图 ...", flush=True)

        txn.commit()
    finally:
        env.close()

    n = write_index(out_index, new_records)
    print()
    print(f"完成：{n} 个 patch → {out_lmdb}")
    print(f"索引：{out_index}")
    if skipped:
        print(f"跳过：{skipped} 张图（见上面的日志）")
    print()
    def rel(p: Path) -> str:
        try:
            return p.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(p)

    print("在 configs/base.yaml 里改成：")
    print("  backend: lmdb")
    print(f"  {'lmdb_path' if 'train' in out_lmdb.name else 'val_lmdb_path'}: "
          f"{rel(out_lmdb)}")
    print(f"  {'train_index' if 'train' in out_lmdb.name else 'val_index'}: "
          f"{rel(out_index)}")
    print("\n（train 和 val 要各自打包成一个库，两个字段都要填。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
