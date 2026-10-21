"""物理噪声合成（线性 RAW 域）。

噪声模型
--------
    σ²[DN] = DN / K + σ_read²

    K        : 系统增益 (e-/DN)  —— 由【平场 + 光子转移曲线】标定
    σ_read   : 读出噪声 (DN)      —— 由【暗帧】标定

    第一项 DN/K 是**光子散粒噪声**，与信号强度相关，亮部主导。
    第二项 σ_read² 是**读出噪声**，与信号无关，是暗部噪声的主因。
    两项合起来才能同时刻画亮部和暗部的噪声 —— 这也是为什么简单加一个
    固定方差的高斯噪声在真实照片上一定会失效。

⚠️ 必须在【线性域】施加（规划文档坑 #2）。
   γ 曲线会扭曲噪声分布（暗部放大、高光压缩），在 sRGB 图上加高斯噪声，
   训练出来的模型泛化到真实噪点图会崩。

标定数据从哪来
--------------
    暗帧   → σ_read（每档 ISO）
    平场   → K（每档 ISO）
    拍法见《数据拍摄清单》第二、三节。
    本文件不负责标定，只负责在给定 K / σ_read 后合成噪声。

----------------------------------------------------------------------
光有 K / σ_read 还不够 —— 实测发现的三处偏差（2026-10-13）
----------------------------------------------------------------------
只按 σ² = DN/K + σ_read² 加**逐像素独立**的高斯噪声，和真实高感噪点
**看起来完全不一样**。用同一台 A7M3 拍 ISO100 + ISO51200 两组 RAW，
量化对比后差在这三处：

======================  ==========  ==========  =============================
指标                    真实        合成        原因
======================  ==========  ==========  =============================
lag-1 自相关             +0.547      +0.000      噪声在**拜耳阵列**上就存在，
                                                经去马赛克后空间相关
通道间相关               +0.198      +0.000      同一个像素只有一个颜色，
                                                邻居是插值出来的
标准差的比值             1.56×       1.00×       白平衡把 R/B 通道整体放大
                                                （索尼实测 R×2.60 / B×1.73），
                                                噪声跟着一起放大
======================  ==========  ==========  =============================

**后果很严重**：模型学会了"去掉白噪声"，遇到成块的、彩色的真实噪点就
认不出来 —— 实测在真实 ISO51200 上只降了 10% 的高频，而同样的图在
合成噪声上能降到底。所以这三个参数必须跟上。

对应的三个入参（都有默认值，不传就是旧行为）：
    wb_gains       每通道噪声放大倍数，来自解码时的白平衡
    corr_lag1      目标 lag-1 空间自相关（实测真实噪声 ≈ 0.547）
    cross_channel  目标通道间相关（实测 ≈ 0.198）
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["DEFAULT_MAX_DN", "add_raw_noise", "NoiseParams"]

# 14-bit RAW 满量程。线性图归一化到 [0,1] 后，1.0 对应这个值。
DEFAULT_MAX_DN = 16383.0


# --------------------------------------------------------------------------- #
# 空间相关噪声：把"去马赛克造成的成块噪声"拟合出来
# --------------------------------------------------------------------------- #
def _corr_kernel_3tap(lag1: float) -> Tuple[float, float]:
    """求对称 3 抽头核 [a, b, a]（2a+b=1），使滤波后白噪声的 lag-1 自相关 = lag1。

    滤波输出的自相关等于核的自相关，所以

        ρ1 = 2ab / (2a² + b²)

    代入 b = 1-2a 整理成 a 的二次方程：

        (6ρ1 + 4)·a² − (4ρ1 + 2)·a + ρ1 = 0

    两个根给出同一条 ρ1，取**中心权重 b 更大**的那个（更像常规低通）。

    ⚠️ 这是对去马赛克空间相关的**近似**，不是精确复现 —— 只拟合了 lag-1。
    好处是零调参：把实测出来的 ρ1 直接填进去就行。
    """
    if lag1 <= 0.0:
        return 0.0, 1.0
    lag1 = min(float(lag1), 0.9)
    qa = 6.0 * lag1 + 4.0
    qb = 4.0 * lag1 + 2.0
    disc = qb * qb - 4.0 * qa * lag1
    if disc < 0.0:
        return 0.0, 1.0
    r = math.sqrt(disc)
    a1 = (qb + r) / (2.0 * qa)
    a2 = (qb - r) / (2.0 * qa)
    a = a1 if (1.0 - 2.0 * a1) >= (1.0 - 2.0 * a2) else a2
    return float(a), float(1.0 - 2.0 * a)


def _filter_sep3(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """沿 H、W 各做一次对称 3 抽头滤波 [a,b,a]，reflect 边界。"""
    if a == 0.0:
        return x
    y = np.pad(x, ((1, 1), (0, 0), (0, 0)), mode="reflect")
    y = a * y[:-2] + b * y[1:-1] + a * y[2:]
    y = np.pad(y, ((0, 0), (1, 1), (0, 0)), mode="reflect")
    return a * y[:, :-2] + b * y[:, 1:-1] + a * y[:, 2:]


def add_raw_noise(
    linear_rgb: np.ndarray,
    K: float,
    sigma_read: float,
    rng: np.random.Generator,
    max_dn: float = DEFAULT_MAX_DN,
    wb_gains: Optional[Sequence[float]] = None,
    corr_lag1: float = 0.0,
    cross_channel: float = 0.0,
) -> np.ndarray:
    """在线性域给干净图加物理噪声。

    Args:
        linear_rgb: (H,W,3) float32，归一化到 [0,1]
        K: 系统增益 (e-/DN)，必须 > 0
        sigma_read: 读出噪声 (DN)，>= 0
        rng: numpy Generator（由调用方持有，保证可复现）
        max_dn: RAW 满量程
        wb_gains: 每通道噪声放大倍数 (gR, gG, gB)。默认 None = 不放大（旧行为）。
            **强烈建议填**：相机解码时的白平衡会把 R/B 通道整体乘 2~3 倍，
            噪声跟着一起被放大。不填的话 R/B 的噪声会明显偏小 ——
            实测索尼 A7M3 差到 1.56 倍。
        corr_lag1: 目标 lag-1 空间自相关。0 = 逐像素独立白噪声。
        cross_channel: 目标通道间相关。0 = R/G/B 各加各的。

    Returns:
        同形状 float32，已 clip 到 [0,1]
    """
    if K <= 0:
        raise ValueError(f"K 必须为正（单位 e-/DN），收到 {K}")
    if sigma_read < 0:
        raise ValueError(f"sigma_read 不能为负，收到 {sigma_read}")

    dn = np.clip(linear_rgb, 0.0, 1.0).astype(np.float32) * max_dn

    g = np.ones(3, dtype=np.float32)
    if wb_gains is not None:
        g = np.asarray(wb_gains, dtype=np.float32).reshape(3)
        if not np.all(g > 0):
            raise ValueError(f"wb_gains 必须都是正数，收到 {wb_gains}")

    # ★ 白平衡修正。
    #   K / σ_read 标定的是**白平衡之前**的传感器 DN，而 dn 是**之后**的。
    #   设通道增益 g，则白平衡前的 DN = dn/g，代进噪声模型再乘回 g²：
    #       σ_pre²  = (dn/g)/K + σ_read²
    #       σ_post² = g²·σ_pre² = g·dn/K + (g·σ_read)²
    #   即散粒项乘 g、读出项乘 g²。g=1 时退化回 σ² = dn/K + σ_read²（原行为）。
    var = dn * g / float(K) + (g * float(sigma_read)) ** 2
    np.maximum(var, 0.0, out=var)
    sigma = np.sqrt(var).astype(np.float32)

    if corr_lag1 > 0.0 or cross_channel > 0.0:
        base = rng.standard_normal(dn.shape, dtype=np.float32)

        if cross_channel > 0.0:
            # noise = sqrt(1-ρ)·独立 + sqrt(ρ)·公共 —— 通道间相关正好是 ρ
            rho = float(min(max(cross_channel, 0.0), 0.95))
            common = rng.standard_normal(dn.shape[:2] + (1,), dtype=np.float32)
            base = np.sqrt(1.0 - rho) * base + np.sqrt(rho) * common

        a, b = _corr_kernel_3tap(corr_lag1)
        base = _filter_sep3(base, a, b)

        # 滤波会改变方差 —— 逐通道重新归一化回目标 σ。
        # 逐通道乘正数不改变通道之间的相关系数，所以这一步不影响上面的 ρ。
        s = base.std(axis=(0, 1), keepdims=True)
        noise = base / np.maximum(s, 1e-12) * sigma
    else:
        noise = rng.standard_normal(dn.shape, dtype=np.float32) * sigma

    return np.clip((dn + noise) / max_dn, 0.0, 1.0)


class NoiseParams:
    """按 (相机, ISO) 查询标定出的 K / σ_read。

    文件格式（noise_params.json）::

        {
          "cameras": ["Canon5D3", "NikonZ6II", "SonyA7M3"],
          "params": {
            "Canon5D3": {"100": [3.20, 2.90], "800": [0.40, 9.00], ...},
            ...
          }
        }

    其中每个值是 ``[K, sigma_read]``。**顺序是固定的**，别写反。
    """

    def __init__(self, cameras: Sequence[str], table: Dict[str, Dict]):
        self.cameras: List[str] = list(cameras)
        if not self.cameras:
            raise ValueError("cameras 不能为空")
        self.table: Dict[str, Dict[int, Tuple[float, float]]] = {
            cam: {int(iso): (float(v[0]), float(v[1])) for iso, v in per_cam.items()}
            for cam, per_cam in table.items()
        }
        missing = [c for c in self.cameras if c not in self.table]
        if missing:
            raise ValueError(f"这些相机在 params 里没有标定数据: {missing}")
        self.validate()

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "NoiseParams":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到噪声标定文件 {path}。"
                "先用《数据拍摄清单》的方法标定，或先用仓库里自带的占位值跑通流程。"
            )
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw["cameras"], raw["params"])

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    def cam_id(self, name: str) -> int:
        try:
            return self.cameras.index(name)
        except ValueError as exc:
            raise KeyError(f"未知相机 {name!r}，已注册的有 {self.cameras}") from exc

    def get(self, cam_id: int, iso: int) -> Tuple[float, float]:
        cam = self.cameras[int(cam_id)]
        per_cam = self.table[cam]
        if int(iso) not in per_cam:
            raise KeyError(
                f"{cam} 没有 ISO {iso} 的标定值，可选: {sorted(per_cam)}"
            )
        return per_cam[int(iso)]

    def iso_list(self, cam_id: int) -> List[int]:
        return sorted(self.table[self.cameras[int(cam_id)]].keys())

    def all_isos(self) -> List[int]:
        """所有相机出现过的 ISO 并集。"""
        return sorted({iso for per_cam in self.table.values() for iso in per_cam})

    def common_isos(self) -> List[int]:
        """所有相机**都有标定**的 ISO 交集。

        验证集应该用这个：保证每个 ISO 档在三台机器上都被评到，
        跨实验的 PSNR 才可比。
        """
        sets = [set(per_cam) for per_cam in self.table.values()]
        return sorted(set.intersection(*sets)) if sets else []

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """标定值合理性检查。**换真实标定数据后务必先跑一次。**"""
        problems: List[str] = []
        for cam, per_cam in self.table.items():
            if not per_cam:
                problems.append(f"{cam}: 没有任何 ISO 标定值")
            for iso, (K, sigma_read) in sorted(per_cam.items()):
                if K <= 0:
                    problems.append(f"{cam} ISO{iso}: K={K} 必须为正")
                if sigma_read < 0:
                    problems.append(f"{cam} ISO{iso}: sigma_read={sigma_read} 不能为负")
                # 经验检查：K 应当随 ISO 上升而单调下降
            isos = sorted(per_cam)
            ks = [per_cam[i][0] for i in isos]
            for a, b, iso_a, iso_b in zip(ks, ks[1:], isos, isos[1:]):
                if b > a * 1.05:
                    problems.append(
                        f"{cam}: K 在 ISO{iso_a}→ISO{iso_b} 上升了 ({a:.4f}→{b:.4f})，"
                        "正常情况下 K 应随 ISO 单调下降，怀疑标定写反了"
                    )
        if problems:
            raise ValueError("noise_params 校验失败:\n  - " + "\n  - ".join(problems))

    def summary(self) -> str:
        lines = [f"NoiseParams: {len(self.cameras)} 台相机"]
        for cam in self.cameras:
            isos = self.iso_list(self.cam_id(cam))
            rng = f"{isos[0]}..{isos[-1]}" if isos else "空"
            lines.append(f"  {cam:<12} {len(isos):>2} 档 ISO  ({rng})")
        return "\n".join(lines)
