"""推理入口：单图 / 批量降噪，含 tiling。

**这里是 C 和 E 的接口交界点。** D21 之后签名冻结，Demo 直接调 `denoise_tiled`。

为什么必须有 tiling
-------------------
5D3 是 2230 万像素，整图直接塞进模型必爆 8GB 显存。分块推理本身不难，
难的是**接缝**：不加处理的话块边界会出现可见的方格。用汉宁窗在重叠区做
加权融合就能无缝拼接（块中心权重高、边缘权重低，重叠区自然过渡）。

用法::

    python infer.py --ckpt runs/exp/last.pth --input photo.jpg \\
                    --camera NikonZ6II --iso 6400 --out clean.png

    python infer.py --ckpt runs/exp/last.pth --input ./photos --out ./results
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# ⚠️ 必须在 import torch 之前设置
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import PROJECT_ROOT, load_config  # noqa: E402
from data.imageio import read_linear, write_linear  # noqa: E402
from data.noise_model import NoiseParams  # noqa: E402
from models import build_model  # noqa: E402
from models.conditioner import infer_noise_params, resolve_noise_params  # noqa: E402
from utils.train_utils import load_checkpoint  # noqa: E402

cv2.setNumThreads(0)

# 4 级下采样（PixelUnshuffle x4），输入边长必须是 16 的倍数
SIZE_MULTIPLE = 16

# EXIF 机型字符串 → 项目内的相机名。
# 各家写法差别很大（尼康是 "NIKON Z 6_2"、索尼是 "ILCE-7M3"），只能靠关键词匹配。
EXIF_CAMERA_HINTS = [
    (("5d mark iii", "5d3", "eos 5d mark iii"), "Canon5D3"),
    (("z 6_2", "z6_2", "z 6ii", "z6ii", "z 6 ii"), "NikonZ6II"),
    (("ilce-7m3", "a7 iii", "a7iii", "a7m3"), "SonyA7M3"),
]

# 品牌级兜底：型号认不出来、但品牌字串认得出来时用（例："Canon EOS R5"）。
# 返回的是**品牌名**，不是标定表里的相机名 —— `infer_noise_params` 见到
# 品牌名会取**同品牌已标定那台**的参数。
# ⚠️ 这是近似：那台机器我们没标定过，K/σ 一定有偏差。UI 必须标明。
EXIF_BRAND_HINTS = [("canon", "Canon"), ("nikon", "Nikon"), ("sony", "Sony")]


# --------------------------------------------------------------------------- #
# 图像读写
# --------------------------------------------------------------------------- #
def read_image(path: Path) -> np.ndarray:
    """读图 → RGB float32 [0,1]。支持 8-bit / 16-bit / 灰度 / TIFF。

    通道顺序和位深的处理统一在 data/imageio.py，别在这里另写一份。
    """
    return read_linear(path)


def write_image(path: Path, rgb: np.ndarray, bit_depth: int = 8) -> None:
    """写图。

    bit_depth=16 时输出 16-bit（TIFF 会走 Deflate 压缩，体积≈PNG）。
    **要在 Photoshop 里继续处理的图，用 16 位** —— 8 位会在暗部丢掉层次，
    而降噪前后最值得看的恰恰就是暗部。
    """
    write_linear(path, rgb, bit_depth=bit_depth)


def write_side_by_side(
    path: Path, tiles: Sequence[tuple[str, np.ndarray]], bit_depth: int = 8
) -> None:
    """把多张图横向拼起来。演示、答辩素材、人工检查都靠它。"""
    arrs = []
    for i, (_, img) in enumerate(tiles):
        if i:
            arrs.append(np.full((img.shape[0], 6, 3), 1.0, dtype=np.float32))
        arrs.append(img.astype(np.float32))
    out = np.concatenate(arrs, axis=1)
    write_image(path, out, bit_depth=bit_depth)


# --------------------------------------------------------------------------- #
# 条件推断
# --------------------------------------------------------------------------- #
def read_exif_camera_iso(path: Path) -> tuple[Optional[str], Optional[int]]:
    """尽力从 EXIF 读机型与 ISO。

    读不到就返回 (None, None)，由命令行参数兜底 —— **不要让 Demo 依赖 EXIF**，
    很多图（截图、社交软件导出的）根本没有 EXIF。
    """
    try:
        from PIL import Image  # Pillow 是可选的
    except ImportError:
        return None, None

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            model = str(exif.get(272, "")) or ""        # 272 = Model
            iso = exif.get(34855)                       # 34855 = ISOSpeedRatings
            if iso is None:
                try:
                    iso = exif.get_ifd(0x8769).get(34855)
                except Exception:
                    iso = None

        cam = None
        low = model.lower()
        for keys, name in EXIF_CAMERA_HINTS:
            if any(k in low for k in keys):
                cam = name
                break
        if cam is None:
            # 型号认不出 → 退一步只认品牌（"Canon EOS R5" → "Canon"）。
            # 返回值是品牌名，下游 infer_noise_params 见到它会用同品牌
            # 已标定机型的参数兜底，而不是三台中位数。
            for key, brand in EXIF_BRAND_HINTS:
                if key in low:
                    cam = brand
                    break
        return cam, (int(iso) if iso else None)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# RAW 输入（双模式）
# --------------------------------------------------------------------------- #
RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".dng", ".raf", ".rw2", ".orf"}


def read_raw_as_srgb(path: Path,
                     wb: Optional[Sequence[float]] = None) -> np.ndarray:
    """RAW → sRGB 显示域 float32 [0,1]，**与训练完全同一条管线**。

        rawpy(postprocess_kwargs) → 线性 [0,1] → apply_srgb_gamma(2.2)

    `wb=None` 用文件内嵌的 as-shot 白平衡；给了具体系数就强制用它。
    前提是演示 RAW 和训练数据是同一个白平衡设置，否则颜色会对不上。

    ⚠️ `data.prepare_data` 是**延迟导入**的 —— 它会把 libraw 拉进进程，
       和 torch + cv2 的 OpenMP 有冲突历史。放在函数里，保证只有真的
       传 RAW 时才付出这个代价。
    """
    from data.imageio import apply_srgb_gamma
    from data.prepare_data import decode_linear

    lin = decode_linear(Path(path), wb if wb else ())
    return apply_srgb_gamma(lin)


def read_raw_camera_iso(path: Path) -> tuple[Optional[str], Optional[int]]:
    """从 RAW 内嵌 EXIF 读 (机型, ISO)。Pillow 读不了 RAW，改用 exifread。

    型号认不出时退到品牌级（"Canon EOS R5" → "Canon"），
    和 `read_exif_camera_iso` 保持同一套降级顺序。
    """
    try:
        from data.prepare_data import camera_from_model, read_exif
    except ImportError:
        return None, None

    try:
        model, iso, _ = read_exif(Path(path))
    except Exception:
        return None, None

    cam = camera_from_model(str(model))
    if cam is None:
        low = str(model).lower()
        for key, brand in EXIF_BRAND_HINTS:
            if key in low:
                cam = brand
                break
    return cam, (int(iso) if iso else None)


# --------------------------------------------------------------------------- #
# Tiling 推理
# --------------------------------------------------------------------------- #
def apply_strength(K: float, sigma_read: float, strength: float) -> tuple:
    """把 (K, σ_read) 往"更吵"的方向推，逼模型降得更狠。

    ``strength = 1.0`` → 原样不动。
    ``strength = s``   → 假装噪声**方差**是原来的 s 倍：

        σ² = DN/K + σ_read²      ⇒      s·σ² = DN/(K/s) + (σ_read·√s)²

    所以只要 ``K → K/s``、``σ → σ·√s``。物理上等价于"把这张图当成高一档
    ISO 拍的"。

    ⚠️ 这不是伪造数据。它是模型的一个**外推旋钮**：
    真实照片的噪声和训练时的合成噪声有域差异（相机的色调曲线、机内降噪、
    JPEG 压缩），实测模型在相机 JPEG 上只降掉约 16% 的高频，明显偏保守。
    这个旋钮就是用来补那一截的 —— 让用户可以选"多干净"还是"多留纹理"。

    返回 ``(K', σ')``。
    """
    s = float(strength)
    if s <= 0:
        raise ValueError(f"strength 必须为正，收到 {s}")
    if s == 1.0:
        return K, sigma_read
    return K / s, sigma_read * math.sqrt(s)


def _pad_to_multiple(x: torch.Tensor, multiple: int = SIZE_MULTIPLE) -> torch.Tensor:
    """把 H/W 补到 multiple 的倍数。reflect 对小图会失败，退回 replicate。"""
    h, w = x.shape[-2:]
    ph = (-h) % multiple
    pw = (-w) % multiple
    if not ph and not pw:
        return x
    mode = "reflect" if (ph < h and pw < w) else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode)


@torch.no_grad()
def denoise_tiled(
    model: torch.nn.Module,
    img_chw: torch.Tensor,
    K: float,
    sigma_read: float,
    tile: int = 512,
    overlap: int = 32,
    device: torch.device | str = "cuda",
    amp_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """分块降噪。

    Args:
        img_chw: (1,3,H,W) float32 [0,1]，CPU 或 GPU 都行
        K: 系统增益 (e-/DN) —— **模型的条件输入，不是相机 ID**
        sigma_read: 读出噪声 (DN)
        tile: 块边长（会被自动对齐到 16 的倍数）
        overlap: 块间重叠像素。约 tile 的 6% 比较平衡

    Returns:
        (1,3,H,W) float32 [0,1]，在 CPU 上

    K / σ_read 怎么来：用 `models.conditioner.infer_noise_params(相机, ISO, 标定表)`。
    没标定过的机型它会自动用中位数兜底 —— 模型依然能用。

    ⚠️ 接口冻结日（D21）之后不要改签名 —— Demo 直接依赖它。
    """
    device = torch.device(device)
    tile = max(SIZE_MULTIPLE, (int(tile) // SIZE_MULTIPLE) * SIZE_MULTIPLE)
    overlap = int(overlap)
    if overlap >= tile:
        raise ValueError(f"overlap({overlap}) 必须小于 tile({tile})")

    if img_chw.dim() != 4 or img_chw.shape[:2] != (1, 3):
        raise ValueError(f"期望 (1,3,H,W)，收到 {tuple(img_chw.shape)}")

    _, _, H, W = img_chw.shape
    k_t = torch.tensor([float(K)], device=device, dtype=torch.float32)
    s_t = torch.tensor([float(sigma_read)], device=device, dtype=torch.float32)

    def _run(patch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device.type == "cuda")):
            return model(patch, k_t, s_t)

    # 小图：单次前向就够了
    if H <= tile and W <= tile:
        x = _pad_to_multiple(img_chw.to(device))
        out = _run(x).float()
        return out[..., :H, :W].clamp(0, 1).cpu()

    stride = tile - overlap
    n_y = max(1, math.ceil((H - overlap) / stride))
    n_x = max(1, math.ceil((W - overlap) / stride))
    pad_h = (n_y - 1) * stride + tile - H
    pad_w = (n_x - 1) * stride + tile - W

    x = img_chw.to(device)
    if pad_h or pad_w:
        mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
        x = F.pad(x, (0, max(0, pad_w), 0, max(0, pad_h)), mode=mode)

    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)

    # 汉宁窗：两端为 0，所以必须 clamp —— 否则重叠区边缘除零出 NaN
    w1 = torch.hann_window(tile, periodic=False, device=device)
    win = (w1[:, None] * w1[None, :]).clamp_min(1e-6)[None, None]

    total = n_y * n_x
    done = 0
    for iy in range(n_y):
        for ix in range(n_x):
            y0, x0 = iy * stride, ix * stride
            patch = x[:, :, y0:y0 + tile, x0:x0 + tile]
            pred = _run(patch).float()
            out[:, :, y0:y0 + tile, x0:x0 + tile] += pred * win
            weight[:, :, y0:y0 + tile, x0:x0 + tile] += win
            done += 1
            if total > 8 and done % max(1, total // 8) == 0:
                print(f"    tiling {done}/{total} 块", flush=True)

    out = out / weight.clamp_min(1e-6)
    return out[..., :H, :W].clamp(0, 1).cpu()


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="用训练好的模型给图片降噪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径")
    ap.add_argument("--input", required=True, help="图片路径或目录")
    ap.add_argument("--out", required=True, help="输出文件或目录")
    ap.add_argument("--camera", default=None,
                    help="相机名（如 NikonZ6II）；不给则尝试读 EXIF。"
                         "**不认识的机型也能用** —— 会自动用同 ISO 的中位数兜底")
    ap.add_argument("--iso", type=int, default=None,
                    help="ISO 值；不给则尝试读 EXIF，再不给用 3200")
    ap.add_argument("--noise-k", type=float, default=None,
                    help="直接指定系统增益 K (e-/DN)，跳过查表。要和 --noise-sigma 一起给")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="直接指定读出噪声 sigma_read (DN)")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="降噪强度。1.0 = 按标定参数原样；>1 更狠（假装噪声方差是 s 倍）；"
                         "<1 更保守。真实相机 JPEG 上建议 1.5~3 之间调")
    ap.add_argument("--blind", action="store_true",
                    help="盲降噪：不告诉模型任何噪声信息。"
                         "⚠️ 当前配置 cond_dropout=0.0，模型训练时没见过这条路，"
                         "输出不可用 —— 只在做消融实验时用，别用它出演示结果")
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--tile", type=int, default=512, help="分块边长（8GB 显存建议 512）")
    ap.add_argument("--overlap", type=int, default=32, help="块间重叠像素")
    ap.add_argument("--compare", action="store_true",
                    help="额外输出 [原图 | 降噪] 对比图")
    ap.add_argument("--bit-depth", choices=["auto", "8", "16"], default="auto",
                    help="输出位深。auto = .tif/.tiff 用 16 位，其余用 8 位。"
                         "要在 Photoshop 里继续处理就用 16")
    return ap.parse_args()


def resolve_bit_depth(requested: str, path: Path) -> int:
    if requested != "auto":
        return int(requested)
    return 16 if path.suffix.lower() in (".tif", ".tiff") else 8


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: 找不到 checkpoint {ckpt_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config") or load_config("configs/base.yaml")
    step = ckpt.get("step", "?")

    # 相机列表从 noise_params 拿，保证与训练时一致
    np_path = PROJECT_ROOT / str(cfg.get("data", {}).get("noise_params", "noise_params.json"))
    noise_params = NoiseParams.load(np_path)
    cameras = noise_params.cameras

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(cfg)
    load_checkpoint(ckpt_path, model, use_ema=bool(ckpt.get("ema")))
    model.eval().to(device)

    amp_dtype = None
    if device.type == "cuda":
        amp_cfg = str(cfg.get("train", {}).get("amp", "bf16")).lower()
        amp_dtype = torch.bfloat16 if amp_cfg in ("bf16", "bfloat16") else None

    print(f"[infer] checkpoint step={step}  相机={cameras}  device={device}  tile={args.tile}")

    in_path = Path(args.input)
    files: list[Path]
    if in_path.is_dir():
        files = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
        )
        if not files:
            print(f"ERROR: 目录里没有图片: {in_path}")
            return 1
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    elif in_path.is_file():
        files = [in_path]
        out_dir = Path(args.out)
        if out_dir.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
            out_dir.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"ERROR: 输入路径不存在: {in_path}")
        return 1

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")

        cam_name, iso_exif = read_exif_camera_iso(f)
        cam_name = args.camera or cam_name
        iso = args.iso or iso_exif or 3200

        # 决定告诉模型多少信息。K < 0 = "不知道噪声水平"，模型自己判断。
        #
        #   三台已标定机型 → 查表给准确的 K / σ_read（常规路径，最准）
        #   其余情况       → 退回盲降噪（PSNR 略低，见文档）
        if args.blind:
            K = sigma_read = -1.0
            note = "盲降噪（强制）"
        elif args.noise_k is not None and args.noise_sigma is not None:
            K, sigma_read = float(args.noise_k), float(args.noise_sigma)
            note = "命令行指定"
        else:
            # ★ 不管认不认识机型，**一律查表给一组参数**。
            #   不认识的机型由 infer_noise_params 用三台相机的**中位数**兜底。
            #
            #   ⚠️ 这里原来会退回"盲降噪"（K = -1），但 `cond_dropout` 默认 0.0 ——
            #      模型训练时**从没见过**"未知"标记，那条路根本没训过。
            #      喂进去的后果是输出≈输入，看起来像"降噪完全没生效"，
            #      而且不报任何错。不认识机型时用中位数参数，
            #      至少落在一个训练过的区域里。
            K, sigma_read, src, used = resolve_noise_params(cam_name, iso, noise_params)
            if src == "calibrated":
                note = f"{cam_name} @ ISO{iso}"
            elif src == "brand":
                note = (f"{cam_name} 型（未标定）→ 按同品牌的 {used} 参数近似"
                        f"（@ISO{iso}，**近似值**）")
            else:
                shown = cam_name or "无 EXIF"
                note = f"{shown} 未标定 → 用三台相机的中位数参数（@ISO{iso}，**近似值**）"

        # 降噪强度旋钮 —— 见 apply_strength()。K < 0 是"盲降噪"标记，不能动。
        if K >= 0 and args.strength != 1.0:
            K, sigma_read = apply_strength(K, sigma_read, args.strength)
            note += f"　强度 ×{args.strength:g}"

        rgb = read_image(f)
        h, w = rgb.shape[:2]
        chw = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
        if K < 0:
            print(f"  {w}x{h}  盲降噪：模型自行判断噪声水平")
        else:
            print(f"  {w}x{h}  噪声 K={K:.5f} e-/DN  σ_read={sigma_read:.2f} DN")
        print(f"    （{note}）")

        pred = denoise_tiled(
            model, chw, K, sigma_read,
            tile=args.tile, overlap=args.overlap,
            device=device, amp_dtype=amp_dtype,
        )

        out_rgb = pred[0].permute(1, 2, 0).numpy()
        if in_path.is_dir() or out_dir.is_dir():
            # 位深决定扩展名：16 位配 TIFF（PS 里更顺手），8 位配 PNG
            ext = ".tif" if args.bit_depth == "16" else ".png"
            dst = out_dir / f"{f.stem}_denoised{ext}"
        else:
            dst = out_dir

        bits = resolve_bit_depth(args.bit_depth, dst)
        write_image(dst, out_rgb, bit_depth=bits)
        print(f"  -> {dst}  ({bits}-bit)")

        if args.compare:
            cmp_ext = ".tif" if bits == 16 else ".png"
            cmp_path = dst.with_name(f"{dst.stem}_compare{cmp_ext}")
            write_side_by_side(
                cmp_path, [("input", rgb), ("denoised", out_rgb)], bit_depth=bits
            )
            print(f"  -> {cmp_path}  ({bits}-bit)")

    print("[infer] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
