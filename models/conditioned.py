"""带噪声条件的 Restormer（不依赖相机 ID）。

**为什么是继承 + 重写 forward，而不是复制一份改**：

    官方主干的所有子模块（patch_embed / encoder_level* / down* / latent /
    up* / reduce_chan* / decoder_level* / refinement / output）都原样保留，
    参数名与官方 checkpoint 逐字一致 → 预训练权重能直接加载。

    新增的只有 conditioner / cond_bias / film 三组参数，命名前缀互不冲突，
    所以 load_state_dict 时被自动跳过，不会报形状不匹配。

**条件输入（2026-10 起）：噪声参数，不是相机 ID**

    forward(img, K, sigma_read)
              K         系统增益 (e-/DN)
              sigma_read 读出噪声 (DN)

    这两个数就是噪声的完整物理描述：测出它们，就唯一确定了「这台机器在这一档
    ISO 下有多吵」。三台相机 × 10 档 ISO = 30 组特性，模型全靠它区分开。
    详见 conditioner.py 的模块说明。

**条件注入的两个位置**（都刻意避开了会对 backbone 权重产生形状依赖的做法）：

    1. patch_embed 之后加一个由噪声参数生成的 per-channel bias。
       —— 不用"把条件拼成额外输入通道"，因为那会把 patch_embed 的
          in_channels 从 3 改成若干，预训练权重当场作废。
    2. 每级 encoder 的第一个 block 之后做 FiLM 调制。
       —— 只插一次。每级 4~8 个 block 全插会推高显存和参数量，收益很小。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conditioner import FiLM, NoiseConditionEmbed
from .restormer_arch import Restormer

__all__ = ["CondRestormer"]

# 条件模块的参数名前缀。加载预训练权重时这些会被跳过。
COND_PREFIXES = ("conditioner.", "cond_bias.", "film.")


class CondRestormer(Restormer):
    """Restormer + 噪声参数条件嵌入（不依赖相机 ID）。

    参数全部走关键字传参，因为父类 __init__ 的参数很多且带默认值。
    backbone 参数（dim / num_blocks / heads / ...）原样透传给 Restormer。
    """

    def __init__(
        self,
        *,
        cond_dim: int = 128,
        cond_hidden: int = 128,
        use_film: bool = True,
        use_cond_bias: bool = True,
        grad_checkpoint: bool = False,
        **backbone_kwargs,
    ):
        super().__init__(**backbone_kwargs)

        # 从已建好的模块里反推 dim，比从 kwargs 取更稳（防止调用方位置传参）
        dim = self.patch_embed.proj.out_channels

        self.cond_dim = cond_dim
        self.use_film = bool(use_film)
        self.use_cond_bias = bool(use_cond_bias)
        self.grad_checkpoint = bool(grad_checkpoint)

        need_cond = self.use_film or self.use_cond_bias
        self.conditioner = (
            NoiseConditionEmbed(cond_dim, cond_hidden) if need_cond else None
        )

        # (1) patch_embed 之后的全局条件 bias
        if self.use_cond_bias:
            self.cond_bias = nn.Linear(cond_dim, dim)
            nn.init.zeros_(self.cond_bias.weight)
            nn.init.zeros_(self.cond_bias.bias)
        else:
            self.cond_bias = None

        # (2) 每级 encoder 第一个 block 之后的 FiLM
        #     四个 encoder 级的通道数依次是 dim, 2dim, 4dim, 8dim
        level_dims = [dim * 2**i for i in range(4)]
        self.film = nn.ModuleList(
            [FiLM(cond_dim, c) for c in level_dims] if self.use_film else []
        )

    # ------------------------------------------------------------------ #
    def _condition(self, x: torch.Tensor, K, sigma_read):
        if self.conditioner is None:
            return None

        batch = x.shape[0]
        if K is None:
            # 整个 batch 走"未知"路径：不告诉它任何噪声信息，它自己从图里判断。
            # 正常推理不会走这里（我们总是查得到标定值），留着是为了消融。
            K = torch.full((batch,), -1.0, device=x.device)
            sigma_read = torch.full((batch,), -1.0, device=x.device)
        else:
            K = torch.as_tensor(K, device=x.device).reshape(-1).float()
            sigma_read = torch.as_tensor(sigma_read, device=x.device).reshape(-1).float()
            if K.shape[0] == 1 and batch > 1:
                K = K.expand(batch)
                sigma_read = sigma_read.expand(batch)
            if K.shape[0] != batch:
                raise ValueError(
                    f"K 的 batch({K.shape[0]}) 与输入({batch}) 不一致"
                )
        return self.conditioner(K, sigma_read).to(x.dtype)

    def _run_level(self, blocks: nn.Sequential, x: torch.Tensor, level: int, cond):
        """跑一级的所有 block，并在第一个 block 之后插一次 FiLM。"""
        film = self.film[level] if (self.use_film and cond is not None) else None
        for i, blk in enumerate(blocks):
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i == 0 and film is not None:
                x = film(x, cond)
        return x

    # ------------------------------------------------------------------ #
    def forward(
        self,
        inp_img: torch.Tensor,
        K=None,
        sigma_read=None,
    ) -> torch.Tensor:
        """inp_img: (B,3,H,W) float32 [0,1]

        K / σ_read: 标量或 (B,) 张量，描述这张图的噪声。
                    **可以传 None** —— 表示"不知道噪声水平"，模型自己判断。
                    也可以传负数（K < 0）来逐样本标记"未知"。

        返回同样形状的降噪结果。全局残差 `+ inp_img` 已包含在内，
        即模型预测的是"噪声残差"，输出直接就是干净图。

        用法：
            model(img, K, sigma)            # 常规路径：EXIF + 标定表给出参数
            model(img)                      # 盲降噪（消融用，PSNR 略低）
        """
        cond = self._condition(inp_img, K, sigma_read)

        inp_enc_level1 = self.patch_embed(inp_img)
        if cond is not None and self.cond_bias is not None:
            inp_enc_level1 = inp_enc_level1 + self.cond_bias(cond)[:, :, None, None]
        out_enc_level1 = self._run_level(self.encoder_level1, inp_enc_level1, 0, cond)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self._run_level(self.encoder_level2, inp_enc_level2, 1, cond)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self._run_level(self.encoder_level3, inp_enc_level3, 2, cond)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self._run_level(self.latent, inp_enc_level4, 3, cond)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
