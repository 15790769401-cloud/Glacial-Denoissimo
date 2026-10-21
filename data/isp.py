"""线性 RAW 域 → sRGB 的简化 ISP。

**为什么单独抠成一个模块**：规划文档坑 #3 —— GT 和 noisy 必须走完全相同的
一套解码参数。把参数冻结在 `ISPParams` 里、`ISP.apply_np` 做成纯函数（无隐藏
状态、无逐图重算），就从代码结构上杜绝了"两条支路参数不一致"这类 bug。

**为什么不用完整 ISP（去马赛克等）**：按《架构与Infra设计》§2.2 的设计，
去马赛克在【离线】阶段用 rawpy 做完，存进数据集的就是已经去马赛克、已白平衡
的线性 RGB。在线阶段只剩「色彩矩阵 + 色调曲线/gamma」，正好是一张 LUT 的量级，
每个 step 都能跑，成本可以忽略。

**色彩矩阵默认是单位阵的正确性说明**：推荐用 rawpy 的
`output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1)` 解码，此时输出的已经是
**线性 sRGB**，色彩转换已经在 rawpy 内部完成了，这里再用单位阵是对的。
只有当你自己在做相机→sRGB 的转换时，才需要把标定得到的 3x3 矩阵填进来。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ISPParams", "ISP", "identity_matrix", "build_tone_lut", "LUT_SIZE"]

LUT_SIZE = 4096


@dataclass(frozen=True)
class ISPParams:
    """一套固化的 ISP 参数。整份数据集共用同一份，不许逐图重算。"""

    color_matrix: np.ndarray  # (3,3) 线性相机RGB → 线性sRGB
    gamma: float = 2.2
    lut_size: int = LUT_SIZE

    def __post_init__(self):
        m = np.asarray(self.color_matrix, dtype=np.float32)
        if m.shape != (3, 3):
            raise ValueError(f"color_matrix 必须是 3x3，收到 {m.shape}")
        object.__setattr__(self, "color_matrix", np.ascontiguousarray(m))


def identity_matrix() -> np.ndarray:
    """线性 sRGB 输入时的默认色彩矩阵。"""
    return np.eye(3, dtype=np.float32)


def build_tone_lut(gamma: float = 2.2, size: int = LUT_SIZE) -> np.ndarray:
    """色调曲线 + gamma 合并成一张查找表。"""
    if size < 2:
        raise ValueError(f"LUT 尺寸太小: {size}")
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.power(x, 1.0 / float(gamma)).astype(np.float32)


class ISP:
    """无状态 ISP。

    同一个实例必须同时用于 GT 和 noisy 两条支路 —— 这正是它存在的意义。
    """

    def __init__(self, params: ISPParams):
        self.params = params
        self.lut = build_tone_lut(params.gamma, params.lut_size)
        # 预转置，省掉每次 forward 的转置开销
        self._matrix_t = np.ascontiguousarray(params.color_matrix.T)

    def apply_np(self, linear_hwc: np.ndarray) -> np.ndarray:
        """linear_hwc: (H,W,3) float32 [0,1] → sRGB (H,W,3) float32 [0,1]"""
        if linear_hwc.ndim != 3 or linear_hwc.shape[2] != 3:
            raise ValueError(f"期望 (H,W,3)，收到 {linear_hwc.shape}")

        h, w, _ = linear_hwc.shape
        if linear_hwc.dtype != np.float32:
            linear_hwc = linear_hwc.astype(np.float32)

        x = np.clip(linear_hwc, 0.0, 1.0).reshape(-1, 3) @ self._matrix_t
        np.clip(x, 0.0, 1.0, out=x)

        idx = (x * (self.lut.size - 1)).astype(np.int32)
        return self.lut[idx].reshape(h, w, 3)

    def __repr__(self) -> str:  # pragma: no cover - 只为日志好看
        m = self.params.color_matrix
        tag = "identity" if np.allclose(m, np.eye(3)) else f"custom{m.shape}"
        return f"ISP(gamma={self.params.gamma}, matrix={tag})"
