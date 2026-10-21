"""模型组装 + 官方预训练权重加载。

⚠️ 这是全项目**最关键的 30 行代码**。

整个"用 sRGB 域"的技术决策，最大的红利就是能直接微调官方 SIDD 权重。
如果下面的 `load_restormer_pretrained` 加载率偏低，说明主干配置被动过
（最常见的原因：改了 dim 通道数），当天的第一优先级是修它，
不要带着随机初始化的模型往下训 —— 那等于把项目最大的优势丢掉了。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .conditioned import COND_PREFIXES, CondRestormer

__all__ = ["build_model", "load_restormer_pretrained", "count_parameters"]

# 属于 Transformer block 的 key 前缀。用于区分"白带的 block"和"关键层"。
BLOCK_PREFIXES = ("encoder_level", "decoder_level", "latent.", "refinement.")


def build_model(cfg: dict) -> CondRestormer:
    """按配置构建模型。

    cfg 里 model 段的字段（dim / num_blocks / num_heads / ...）直接透传给
    CondRestormer，其中条件相关的字段单独摘出来。

    注意：**不需要相机数量** —— 条件输入是噪声参数，不是相机 ID，
    所以模型天然地对未见过的机身有效。见 models/conditioner.py。
    """
    m = dict(cfg.get("model", cfg))
    return CondRestormer(
        cond_dim=m.pop("cond_dim", 128),
        cond_hidden=m.pop("cond_hidden", 128),
        use_film=m.pop("use_film", True),
        use_cond_bias=m.pop("use_cond_bias", True),
        grad_checkpoint=m.pop("grad_checkpoint", False),
        # 其余字段是 backbone 参数
        inp_channels=m.pop("inp_channels", 3),
        out_channels=m.pop("out_channels", 3),
        dim=m.pop("dim", 48),
        num_blocks=list(m.pop("num_blocks", [2, 3, 3, 4])),
        num_refinement_blocks=m.pop("num_refinement_blocks", 4),
        heads=list(m.pop("num_heads", [1, 2, 4, 8])),
        ffn_expansion_factor=m.pop("ffn_expansion_factor", 2.66),
        bias=m.pop("bias", False),
        LayerNorm_type=m.pop("layer_norm_type", "WithBias"),
    )


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """总参数 / backbone 参数 / 条件模块参数，分开统计。

    条件模块的参数量占比应当很小（<1%）。如果超过 5%，说明 cond_dim 开太大了。
    """
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(
        p.numel()
        for k, p in model.named_parameters()
        if not k.startswith(COND_PREFIXES)
    )
    return {"total": total, "backbone": backbone, "conditioning": total - backbone}


def load_restormer_pretrained(
    model: torch.nn.Module,
    ckpt_path: str,
    verbose: bool = True,
) -> Tuple[list, list]:
    """加载官方 Restormer 权重，**自动跳过形状不匹配的层**。

    兼容官方 checkpoint 的三种常见形态：
        {'params': sd} / {'state_dict': sd} / sd 本身
    以及 DataParallel 留下的 'module.' 前缀。

    返回 (loaded_keys, skipped_keys)。

    预期结果：本项目的裁剪方式（保持通道数、只砍 block 数）下，
    backbone 覆盖率应当接近 100% —— 因为裁剪后的模型是原版的**子集**，
    每个 block 都能在 checkpoint 里找到同名同形状的对应项。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("params", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    sd = {
        k.removeprefix("module."): v
        for k, v in ckpt.items()
        if isinstance(v, torch.Tensor)
    }
    if not sd:
        raise ValueError(f"从 {ckpt_path} 里没读到任何张量权重，检查 checkpoint 格式")

    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    filled = set(loaded)
    backbone_keys = [k for k in own if not k.startswith(COND_PREFIXES)]
    # ★ 真正要紧的是【我们自己的哪些参数没填上】，而不是 checkpoint 里哪些没用上。
    #   前者会保持随机初始化、拖累效果；后者只是白带的，无害。
    unfilled = [k for k in backbone_keys if k not in filled]

    model.load_state_dict(own, strict=True)

    if verbose:
        ratio = len(loaded) / max(1, len(backbone_keys))
        print(f"[pretrained] checkpoint 张量 {len(sd)} 个 → 用上 {len(loaded)}，"
              f"未用上 {len(skipped)}")
        print(f"[pretrained] backbone 覆盖率 {ratio:.1%}  "
              f"({len(loaded)}/{len(backbone_keys)})")

        if unfilled:
            by_block: Dict[str, int] = {}
            critical: List[str] = []
            for k in unfilled:
                if k.startswith(BLOCK_PREFIXES):
                    parts = k.split(".")
                    key = ".".join(parts[:2]) if len(parts) > 1 else k
                    by_block[key] = by_block.get(key, 0) + 1
                else:
                    critical.append(k)

            print(f"[pretrained] ⚠️  本模型有 {len(unfilled)} 个参数没填上（保持随机初始化）：")
            for blk, n in sorted(by_block.items()):
                print(f"[pretrained]      {blk:<24} {n:>3} 个")
            if critical:
                print(f"[pretrained]      ⚠️ 其中 {len(critical)} 个是【非 block 的关键层】，"
                      "必须查清楚：")
                for k in critical[:12]:
                    print(f"[pretrained]         ✗ {k}")
            else:
                print("[pretrained]      全部属于 Transformer block。若这些 block "
                      "在 checkpoint 里本来就不存在（我们比它多），属预期；")
                print("[pretrained]      但如果是【同一编号】的 block 没填上，"
                      "说明 checkpoint 比我们小，那部分只能是随机的。")

        if ratio < 0.60:
            print("[pretrained] ⚠️  覆盖率过低！八成是改了 dim / num_heads 通道数。")
            print("[pretrained]    正确做法是保持 dims/heads 不变，只砍 num_blocks。")

    return loaded, skipped
