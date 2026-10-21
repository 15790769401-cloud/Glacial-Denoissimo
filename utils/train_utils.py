"""训练工具：EMA / 调度器 / checkpoint / 日志 / 随机种子。

集中放在一处，是为了让 `train.py` 只剩训练循环本身 —— 5 个人协作时，
训练脚本越短越不容易冲突。
"""

from __future__ import annotations

import contextlib
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed",
    "EMA",
    "build_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "MetricLogger",
    "format_bytes",
    "count_parameters",
]


# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定随机性。

    默认 deterministic=False —— 严格逐位复现会让训练慢 30%，对降噪任务不值得。
    **但验证集必须确定**，那由 dataset 侧的固定 seed 保证，不靠这里。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------- #
class EMA:
    """指数滑动平均（Exponential Moving Average）。

    **几乎总能白拿 0.1~0.3 dB**，显存开销对 14M 参数的模型只有几十 MB。
    评测和 Demo 一律用 EMA 权重。

    用法::

        ema = EMA(model, decay=0.999)
        ...
        ema.update(model)                       # 每个优化步之后
        with ema.average_parameters(model):     # 验证时临时换入
            psnr = evaluate(model, ...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay 应在 (0,1)，收到 {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k in sd:
            sd[k].copy_(self.shadow[k].to(sd[k].dtype))

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[nn.Module]:
        """临时换入 EMA 权重，退出时还原原始权重。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: Mapping[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.float())


# --------------------------------------------------------------------------- #
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    min_lr_ratio: float = 1.0 / 30.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 warmup + cosine 退火到 min_lr_ratio * base_lr。"""
    warmup_iters = max(1, int(warmup_iters))
    total_iters = max(warmup_iters + 1, int(total_iters))

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / warmup_iters
        progress = (step - warmup_iters) / (total_iters - warmup_iters)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    cfg: Optional[Mapping] = None,
    extra: Optional[Mapping] = None,
) -> Path:
    """存 checkpoint。

    **必须把 cfg 一起存进去** —— 一周后没人记得这组超参是什么，
    不存配置的 checkpoint 是没法复现也无法继续训练的。
    """
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
        "config": dict(cfg) if cfg else {},
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(dict(extra))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)
    return p


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    use_ema: bool = False,
) -> Dict[str, Any]:
    """载入 checkpoint。**断点续训必须能用** —— 8GB 卡上训练中断是常态。

    Args:
        use_ema: True 时把 EMA 权重灌进 model（推理/评测用）；
                 False 时保留训练权重，EMA 只恢复影子副本（续训用）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到 checkpoint {p}")

    ckpt = torch.load(p, map_location=map_location)
    if "model" not in ckpt:
        raise ValueError(f"{p} 看起来不是本项目的 checkpoint（缺少 'model' 键）")

    model.load_state_dict(ckpt["model"])

    if use_ema and ema is None and ckpt.get("ema") is not None:
        # ★ 只加载 EMA 权重的路径：调用方（infer.py / 评测 / Demo）只是要拿
        #   EMA 权重推理，并没有建 EMA 对象。EMA.shadow 的 key 与
        #   model.state_dict() 完全一致（见 EMA.__init__），直接灌进去就行。
        #
        #   少了这一条，`use_ema=True` 会**静默失效** —— 推理实际用的是原始
        #   训练权重，而 train.py 的验证 PSNR 用的是 EMA，两边差 0.1~0.3 dB，
        #   于是"演示效果"和"报告数字"对不上，而且完全不报错。
        model.load_state_dict(ckpt["ema"])
    elif ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
        if use_ema:
            ema.copy_to(model)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    return {
        "step": int(ckpt.get("step", 0)),
        "config": ckpt.get("config", {}),
    }


# --------------------------------------------------------------------------- #
class MetricLogger:
    """指标日志。**CSV 永远写，wandb 可选。**

    5 个人并行跑实验时，没有统一日志就等于没做实验。wandb 连不上（校园网常见）
    也不能因此不记录 —— 所以 CSV 是主力，wandb 只是锦上添花。
    """

    def __init__(
        self,
        out_dir: str | Path,
        run_name: str,
        use_wandb: bool = False,
        wandb_cfg: Optional[Mapping] = None,
        wandb_mode: Optional[str] = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.out_dir / f"{run_name}.csv"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fields: list[str] = []
        self.wandb = None

        if use_wandb:
            try:
                import os

                if wandb_mode:
                    os.environ["WANDB_MODE"] = wandb_mode
                import wandb

                self.wandb = wandb.init(
                    project="camera-denoise",
                    name=run_name,
                    config=dict(wandb_cfg) if wandb_cfg else None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[logger] wandb 不可用，仅写 CSV（{type(exc).__name__}: {exc}）")
                self.wandb = None

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        row = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        if not self._fields:
            self._fields = list(row.keys())
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fields)
            self._writer.writeheader()
        for k in row:
            if k not in self._fields:  # 新出现的键，补一列
                self._fields.append(k)
                self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fields)
        self._writer.writerow(row)
        self._csv_file.flush()

        if self.wandb is not None:
            try:
                self.wandb.log({**row, "step": int(step)})
            except Exception:  # noqa: BLE001
                pass

    def log_image(self, key: str, image_uint8: np.ndarray, step: int, caption: str = "") -> None:
        """把验证图传上去。**这条比曲线更有用** —— 肉眼能看出模型是在变好还是糊掉了。"""
        if self.wandb is None:
            return
        try:
            import wandb

            self.wandb.log({key: wandb.Image(image_uint8, caption=caption)}, step=int(step))
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self._csv_file.close()
        finally:
            if self.wandb is not None:
                try:
                    self.wandb.finish()
                except Exception:  # noqa: BLE001
                    pass

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
