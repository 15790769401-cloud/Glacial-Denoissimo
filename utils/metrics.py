"""评测指标：PSNR / SSIM。

口径与 SIDD / DND / NTIRE 一致 —— **在 8-bit sRGB 域、按 [0,255] 量程计算**。
和别人的数字对不上，多半是 data_range 用错了（用了 [0,1] 却报 [0,255] 的 PSNR，
差 48.13 dB）。

SSIM 用 11x11 高斯窗（sigma=1.5）实现，与 scikit-image 默认参数一致，
所以本文件的数字可以直接和论文里的对比。
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

__all__ = ["psnr", "ssim", "psnr_ssim", "DATA_RANGE_8BIT"]

cv2.setNumThreads(0)

DATA_RANGE_8BIT = 255.0


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = DATA_RANGE_8BIT) -> float:
    """pred / target: 同形状数组。默认按 8-bit 量程。"""
    if pred.shape != target.shape:
        raise ValueError(f"形状不一致: {pred.shape} vs {target.shape}")
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0  # 完全一致，避免 log10(0)
    return float(10.0 * np.log10(data_range * data_range / mse))


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    """单通道 SSIM，11x11 高斯窗。x / y 为 float64。"""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    win = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(win, win).astype(np.float64)

    border = dict(borderType=cv2.BORDER_REFLECT)
    mu_x = cv2.filter2D(x, -1, window, **border)[4:-4, 4:-4]
    mu_y = cv2.filter2D(y, -1, window, **border)[4:-4, 4:-4]

    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = cv2.filter2D(x * x, -1, window, **border)[4:-4, 4:-4] - mu_x2
    sig_y2 = cv2.filter2D(y * y, -1, window, **border)[4:-4, 4:-4] - mu_y2
    sig_xy = cv2.filter2D(x * y, -1, window, **border)[4:-4, 4:-4] - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    return float(np.mean(num / den))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = DATA_RANGE_8BIT) -> float:
    """多通道 SSIM（各通道分别算再平均）。

    输入按 [0,255] 量程。若传进来的是 [0,1]，会自动放大到 255，
    避免"数值看着对、和一比差 48 dB"这种低级事故。
    """
    if pred.shape != target.shape:
        raise ValueError(f"形状不一致: {pred.shape} vs {target.shape}")

    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    if data_range == 1.0 or max(float(x.max()), float(y.max())) <= 1.0 + 1e-6:
        x = x * 255.0
        y = y * 255.0

    if x.ndim == 2:
        return _ssim_channel(x, y)
    if x.ndim != 3:
        raise ValueError(f"期望 2D 或 3D 数组，收到 {x.shape}")
    return float(np.mean([_ssim_channel(x[..., c], y[..., c]) for c in range(x.shape[2])]))


def psnr_ssim(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """一次算两个指标，返回 8-bit 口径的结果。"""
    return {
        "psnr": psnr(pred, target, DATA_RANGE_8BIT),
        "ssim": ssim(pred, target, DATA_RANGE_8BIT),
    }
