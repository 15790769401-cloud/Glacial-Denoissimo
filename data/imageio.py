"""统一的 16-bit 图像读写（PNG / TIFF）。

**为什么要把读写收敛到一个模块**：整条链路上最容易出错的两件事 ——
通道顺序（RGB/BGR）和位深 —— 散落在各个文件里就一定会有一处写错，
而且错了还看不出来（图能打开、颜色略偏、精度悄悄掉了）。

写成一份、只测一遍。`selfcheck.py` 里有对应的往返测试。

--------------------------------------------------------------------------
实测结论（opencv 5.0 / Pillow 12.3 / tifffile 2026.3，2026-10 实测）
--------------------------------------------------------------------------

**① cv2 与 Photoshop 互通没问题。**
用 cv2 写入 BGR 数组，落盘的 TIFF 是标准 RGB —— 拿严格按 TIFF 规范实现的
tifffile 读回来，颜色是对的。这条很反直觉，但不实测不敢下结论。

**② 四种压缩全部互通。**
none / LZW / Deflate / PackBits 都能被 cv2 和 tifffile 正确读出。
（注意：对噪声大的图 LZW 反而会**变大**，LZW 适合大片平坦区域。
  对照片用 Deflate，实测体积与 PNG 相当。）

**③ ⚠️ Pillow 会把 16-bit RGB 静默降成 8-bit。**
PIL 打开我们的 16 位图，`np.array(img).dtype` 是 `uint8`，数值直接被
除以 257 砍掉低 8 位 —— **不报错、不警告**。用 PIL 读会被误导，
用 PIL 写会直接毁掉数据。

    → **不要用 PIL 碰这些图的像素。**
    → `infer.py` 里用 PIL 只是为了读 EXIF，不碰像素，是安全的。

--------------------------------------------------------------------------
TIFF vs PNG 怎么选
--------------------------------------------------------------------------

本项目**两者都支持**，按需要选：

| | TIFF | PNG |
|---|---|---|
| Photoshop | ✅ 原生支持，16 位工作流顺畅 | ⚠️ PS 支持 16 位 PNG 但有历史遗留问题 |
| Lightroom / Capture One 导出 | ✅ 原生 | ⚠️ 要绕一下 |
| 体积（24MP 16-bit） | Deflate 压缩后 ≈ 40~70MB | ≈ 40~70MB |
| 体积（不压缩） | **144MB** | — |
| ML 生态 | 略少 | 更多 |

**建议：TIFF + Deflate 压缩。** 体积和 PNG 一样，但 PS 里更顺手。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

cv2.setNumThreads(0)

__all__ = [
    "read_linear",
    "write_linear",
    "write_srgb_preview",
    "apply_srgb_gamma",
    "TIFF_DEFLATE",
    "TIFF_LZW",
    "TIFF_NONE",
    "SUPPORTED_EXT",
]

# cv2.IMWRITE_TIFF_COMPRESSION 的取值
TIFF_NONE = 1
TIFF_LZW = 5
TIFF_DEFLATE = 8        # 即 Photoshop 里的 "ZIP" 压缩
TIFF_PACKBITS = 32773

SUPPORTED_EXT = (".png", ".tif", ".tiff", ".bmp")

_DEFAULT_TIFF_COMPRESSION = TIFF_DEFLATE


# --------------------------------------------------------------------------- #
# 读
# --------------------------------------------------------------------------- #
def read_linear(path: Union[str, Path]) -> np.ndarray:
    """读 8-bit / 16-bit 图像 → (H,W,3) float32 [0,1]。**按 RGB 顺序**。

    支持 .png / .tif / .tiff / .bmp 以及相机可能直接给的 .jpg。
    单通道和 4 通道（BGRA）也会自动转成 3 通道。
    """
    path = Path(path)
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(
            f"读不到图: {path}\n"
            "（确认文件存在；TIFF 若带图层/Alpha/CMYK，OpenCV 可能读不了，"
            "请在 Photoshop 里另存为「拼合、无 Alpha、RGB」的 TIFF）"
        )

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)   # cv2 读出来是 BGR
    else:
        raise ValueError(f"{path.name}: 不支持的通道数 {arr.shape[2]}")

    return to_float01(arr)


def to_float01(arr: np.ndarray) -> np.ndarray:
    """uint8 / uint16 → float32 [0,1]。"""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype in (np.float32, np.float64):
        out = arr.astype(np.float32)
        return out / 255.0 if out.max() > 1.0 + 1e-6 else out
    raise ValueError(f"没见过的 dtype: {arr.dtype}")


# --------------------------------------------------------------------------- #
# 写
# --------------------------------------------------------------------------- #
def _compression_args(path: Path, compression: int) -> list:
    """只有 TIFF 支持压缩参数；其它格式传了 cv2 会忽略，但显式分开更清楚。"""
    if path.suffix.lower() in (".tif", ".tiff"):
        return [cv2.IMWRITE_TIFF_COMPRESSION, int(compression)]
    return []


def write_linear(
    path: Union[str, Path],
    rgb01: np.ndarray,
    compression: int = _DEFAULT_TIFF_COMPRESSION,
    bit_depth: int = 16,
) -> None:
    """把 (H,W,3) float32 [0,1] 写成 16-bit（默认）或 8-bit 图像。

    ⚠️ 写入时会做 BGR 转换 —— cv2 的约定如此，落盘后是标准 RGB。

    Args:
        compression: 仅对 TIFF 生效。默认 Deflate（体积≈PNG）
        bit_depth: 16 或 8。**训练数据一律用 16**，8 位会在暗部把噪声量化掉。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError(f"期望 (H,W,3)，收到 {rgb01.shape}")

    a = np.clip(rgb01, 0.0, 1.0)
    if bit_depth == 16:
        arr = (a * 65535.0).round().astype(np.uint16)
    elif bit_depth == 8:
        arr = (a * 255.0).round().astype(np.uint8)
    else:
        raise ValueError(f"bit_depth 只支持 8 / 16，收到 {bit_depth}")

    bgr = np.ascontiguousarray(arr[:, :, ::-1])
    if not cv2.imwrite(str(path), bgr, _compression_args(path, compression)):
        raise IOError(f"写盘失败: {path}")


def apply_srgb_gamma(rgb01: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """线性 → 显示用 sRGB（近似）。**仅供人眼查看，不要用于训练数据。**"""
    return np.power(np.clip(rgb01, 0.0, 1.0), 1.0 / float(gamma)).astype(np.float32)


def write_srgb_preview(
    path: Union[str, Path],
    linear_rgb01: np.ndarray,
    gamma: float = 2.2,
    bit_depth: int = 16,
) -> None:
    """把【线性】图转成【可以直接在 Photoshop 里看】的 sRGB 图再存。

    ⚠️ 为什么需要这个：数据集里存的是**线性**数据，没有 gamma。
    直接用 PS 打开会又暗又灰 —— 那是**正确**的，不是文件坏了。
    想在 PS 里肉眼看，必须套一次 gamma，这个函数就是干这个的。

    **预览图不要拿去做训练** —— 它已经不是线性数据了。
    """
    write_linear(path, apply_srgb_gamma(linear_rgb01, gamma), bit_depth=bit_depth)
