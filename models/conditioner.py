"""噪声条件嵌入 —— 让一个模型覆盖三台机身 × 全部 10 档 ISO。

设计要点
--------
**不用相机 ID，改用噪声参数。**

原设计是 ``cam_embed[相机] + ISO 编码``。换成
``NoiseMLP([log10 K, log10 σ_read, log10 中灰噪声])`` 的原因：

* **加机器就要动模型** —— 扩 embedding 并重训。用物理参数则只是多两行标定数据
* **信息冗余又混杂** —— 相机 ID 和 ISO 编码高度冗余，却把"哪台机器"和
  "有多吵"混在一个向量里，模型没法把二者拆开
* **不利于消融** —— 噪声参数是物理量，可以直接扫；相机 ID 只是个查表索引

K（系统增益 e-/DN）和 σ_read（读出噪声 DN）是噪声的完整物理描述：
测出这两个数，就唯一确定了「这台机器在这一档 ISO 下有多吵」。
**三台相机 × 10 档 ISO = 30 组特性，全靠这一个模块区分开。**

"未知"模式：留给消融的备用通路
------------------------------
第 4 维是一个 0/1「已知标记」，训练时按 ``data.cond_dropout`` 的概率把前 3 维
清零表示"不知道"。**三机型方案下我们总能从 EXIF + 标定表拿到准确参数，
所以默认 ``cond_dropout = 0.0``** —— 全部样本都带条件，PSNR 最高。
保留这条通路只是为了留住一个消融对比。

为什么"未知"模式代价不大：**条件化提供的信息，模型自己也能从图里估计出来**
（噪声是看得见的），所以它更像"学习捷径"而不是"新信息" ——
常规 ISO 下差距不大，**极端 ISO 下才拉开**。

副作用（好的）：交叉合成
------------------------
不用相机 ID 之后，**可以用 A 相机的干净图 + B 相机的噪声参数**合成训练样本。
物理上完全合法，而且：

* 把有效训练数据放大了 N 倍（N = 3）
* 天然地教模型"噪声只跟噪声参数有关，跟哪台相机拍的无关"
  —— 三台相机之间的泛化就来自这里

见 `data.cross_camera_noise` 配置项。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["NoiseConditionEmbed", "FiLM", "infer_noise_params",
           "resolve_noise_params"]

# 归一化用的对数范围。超出范围会被 clamp，不会崩。
LOG_K_RANGE = (-3.0, 1.5)        # 1e-3 … ~30 e-/DN
LOG_SIGMA_RANGE = (-1.0, 2.5)    # 0.1 … ~300 DN
LOG_LEVEL_RANGE = (-1.0, 3.0)    # 中灰处噪声水平（DN）

MAX_DN = 16383.0                 # 14-bit RAW 满量程
MID_GRAY = 0.18                  # 中灰反射率


def _norm(x_log: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """把对数域的值线性映射到 [-1, 1]，超出范围钳到 [-4, 4]。"""
    return ((x_log - lo) / (hi - lo) * 2.0 - 1.0).clamp(-4.0, 4.0)


class NoiseConditionEmbed(nn.Module):
    """把噪声参数编码成条件向量。

    输入 4 维：``[log10 K, log10 σ_read, log10 中灰噪声水平, 已知标记]``。

    前两维是标定出来的物理参数，第三维由它们推出（直接对应"看起来有多吵"）。
    第 4 维是**已知标记**：``1`` = 参数可信，``0`` = "不知道"（前 3 维归零）。
    训练时按 `data.cond_dropout` 的概率走"不知道"那一路；默认 0.0，
    即全部样本都带条件。

    ⚠️ 第 4 维不能省。否则"不知道"只能靠把参数设成某个具体值来近似，
    模型分不清「我不知道」和「噪声刚好中等」。

    ⚠️ 一个容易想当然的点：条件化提供的信息，模型其实**自己也能从图里
    估计出来**（噪声是看得见的）。所以它更像是"学习捷径"而不是"新信息" ——
    常规 ISO 下差距不大，**极端 ISO（51200）下才拉开**。

    Args:
        dim: 输出条件向量维度
        hidden: MLP 隐藏层宽度
    """

    def __init__(self, dim: int = 128, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # 输入已经是 4 个归一化标量，不需要额外的位置编码；
        # 但要保证初始输出是小的随机值，避免一开始就把特征推偏。
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def features(
        K: torch.Tensor,
        sigma_read: torch.Tensor,
        known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """K、σ_read → (B, 4) 归一化特征。

        第 4 维是"是否已知"的 0/1 标记。**这一维不能省** ——
        否则"未知"只能靠把参数设成某个具体值来近似，模型分不清
        "我不知道"和"噪声刚好中等"。

        `known=False` 的样本，前三维一律置零（模型会忽略它们）。
        """
        K = K.float().reshape(-1)
        s = sigma_read.float().reshape(-1)
        if known is None:
            # 约定：K < 0 表示未知
            known = (K > 0) & (s >= 0)
        known = known.bool().reshape(-1)

        Kc = K.clamp_min(1e-6)
        sc = s.clamp_min(1e-6)
        # 中灰处的总噪声标准差（DN）：σ² = DN/K + σ_read²
        dn = MID_GRAY * MAX_DN
        level = torch.sqrt(dn / Kc + sc * sc)

        nk = _norm(torch.log10(Kc), *LOG_K_RANGE)
        ns = _norm(torch.log10(sc), *LOG_SIGMA_RANGE)
        nl = _norm(torch.log10(level), *LOG_LEVEL_RANGE)
        zero = torch.zeros_like(nk)

        return torch.stack(
            [
                torch.where(known, nk, zero),
                torch.where(known, ns, zero),
                torch.where(known, nl, zero),
                known.float(),
            ],
            dim=-1,
        )

    def forward(self, K: torch.Tensor, sigma_read: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(K, sigma_read))


class FiLM(nn.Module):
    """特征级条件调制:  x = x * (1 + gamma) + beta

    `to_gb` 零初始化 → 初始时 gamma=0, beta=0，整个模块是恒等映射。
    这一点对"加载预训练权重后继续训练"是必须的。
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W), cond: (B,cond_dim) → (B,C,H,W)"""
        gb = self.to_gb(cond).to(x.dtype)
        g, b = gb.chunk(2, dim=-1)
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


# --------------------------------------------------------------------------- #
# 品牌名（小写）。型号认不出来、但认得出品牌时用它兜底。
_BRANDS = ("canon", "nikon", "sony")


def _brand_match(camera, cameras):
    """型号认不出时，按品牌找**同品牌已标定**的那台。找不到返回 None。

    例：EXIF 读到 "Canon EOS R5" → 型号没标定过，但品牌是 canon →
    返回 "Canon5D3"。同品牌传感器的噪声特性更接近，比"三台中位数"合理。
    """
    if not camera:
        return None
    low = str(camera).lower()
    for brand in _BRANDS:
        if low.startswith(brand):
            same = [c for c in cameras if c.lower().startswith(brand)]
            # 同品牌有多个已标定机型时没法确定用哪个，退回中位数
            return same[0] if len(same) == 1 else None
    return None


def resolve_noise_params(camera, iso: int, table):
    """(相机, ISO) → (K, σ_read, source, 实际用的相机名)。

    `source` 有三档，**UI 要据此显示不同的提示**：

        "calibrated"  精确命中已标定的机型           —— 最准
        "brand"       型号未标定，按**同品牌**那台近似 —— 近似，UI 要标明
        "median"      品牌也认不出，用三台的中位数     —— 最粗，UI 要标明

    ⚠️ 后两档都是**近似**，我们没标定过那台机器，(K, σ) 一定有偏差。
       所以**不能**对外声称"支持任意相机" —— 见《交付说明》的口径一节。
    """
    cameras = list(table.cameras)

    def pick_iso(cam: str, want: int) -> int:
        avail = table.iso_list(table.cam_id(cam))
        return min(avail, key=lambda v: abs(math.log2(max(v, 1) / max(want, 1))))

    # ① 精确命中
    if camera and camera in cameras:
        k, s = table.get(table.cam_id(camera), pick_iso(camera, iso))
        return k, s, "calibrated", camera

    # ② 品牌兜底
    m = _brand_match(camera, cameras)
    if m:
        k, s = table.get(table.cam_id(m), pick_iso(m, iso))
        return k, s, "brand", m

    # ③ 中位数兜底
    if not cameras:
        raise ValueError("noise_params 里一台相机都没有")
    ks, ss = [], []
    for cam in cameras:
        k, s = table.get(table.cam_id(cam), pick_iso(cam, iso))
        ks.append(k)
        ss.append(s)
    return (float(torch.tensor(ks).median()), float(torch.tensor(ss).median()),
            "median", None)


def infer_noise_params(
    camera: str | None,
    iso: int,
    table,
    fallback: str = "median",
) -> tuple[float, float]:
    """(相机, ISO) → (K, σ_read)。**模型本身不需要知道相机是什么。**

    分三档兜底：精确机型 → 同品牌 → 三台中位数。细节见 `resolve_noise_params`。

    Args:
        camera: 相机名、品牌名（"Canon"），或 None
        iso: ISO 值。表里没有该档 → 取最接近的一档
        table: `data.noise_model.NoiseParams` 实例
        fallback: 保留参数，目前恒为"中位数"（代码里没有第二个分支）

    Returns:
        (K, sigma_read)
    """
    k, s, _source, _used = resolve_noise_params(camera, iso, table)
    return k, s
