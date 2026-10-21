"""训练数据集：干净线性图 → (noisy, gt) sRGB 训练对。

数据流（见《架构与Infra设计》§2.2）
-----------------------------------
    离线：RAW ──解码+去马赛克+白平衡──→ 线性RGB ──存 16-bit PNG / LMDB

    在线：   ┌─ 干净线性 patch ─┬─→ 加物理噪声 ─→ 上 ISP ─→ noisy ┐
             │                  └─→ 上 ISP ─────────────→ gt    ├→ loss
             └─ K, σ_read（噪声参数，不是相机 ID）────────────────┘

**为什么离线存线性图而不是 sRGB 图**：噪声必须在线性域合成（坑 #2）。
如果离线就上了色调曲线，线性信息永久丢失，物理噪声模型就没法用了。

**为什么同一张干净 patch 可以反复用**：每次采样不同的噪声实现，一张图能生成
无穷多训练样本。这就是"标定 + 合成"路线能替代大规模实拍配对的原因。

索引文件 (jsonl) 格式，每行一条::

    {"src": "data/linear/Z6II_s045.png", "cam": 1, "scene": "Z6II_s045"}
    {"src": "0000123_04", "cam": 1, "scene": "Z6II_s045", "h": 128, "w": 128}

- backend=png  时 ``src`` 是图片路径
- backend=lmdb 时 ``src`` 是 LMDB 的 key，且必须带 ``h`` / ``w``

**场景字段 ``scene`` 是干什么的**：用来划分 train/val。同一张原图裁出的 patch
必须整体落在同一侧，否则数据泄漏、验证 PSNR 虚高好几个 dB。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .imageio import read_linear
from .isp import ISP, ISPParams, identity_matrix
from .noise_model import DEFAULT_MAX_DN, NoiseParams, add_raw_noise

# DataLoader 的多个 worker 各自开 OpenCV 线程会互相抢，反而更慢
cv2.setNumThreads(0)

__all__ = [
    "load_index",
    "write_index",
    "split_by_scene",
    "check_no_scene_leak",
    "DenoiseDataset",
    "build_isp",
    "build_datasets",
    "build_loaders",
]


# --------------------------------------------------------------------------- #
# 索引文件读写
# --------------------------------------------------------------------------- #
def load_index(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到索引 {path}。先跑 `python data/make_dummy_data.py` 生成假数据，"
            "或用 data/build_lmdb.py 从真实数据建索引。"
        )
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
            for key in ("src", "cam", "scene"):
                if key not in rec:
                    raise ValueError(f"{path}:{lineno} 缺少字段 {key!r}")
            records.append(rec)
    if not records:
        raise ValueError(f"索引为空: {path}")
    return records


def write_index(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_by_scene(
    records: Sequence[dict],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[dict], List[dict]]:
    """按**场景**划分 train / val。

    ⚠️ 绝对不要按 patch 随机划分。同一张原图裁出来的 32 个 patch 高度相关，
    随机划分会让验证集里出现训练集的"近亲"，验证 PSNR 虚高好几个 dB，
    而且你会以为模型真的很好 —— 这是本项目最隐蔽的一个坑。
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction 应在 [0,1)，收到 {val_fraction}")

    scenes = sorted({str(r["scene"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(scenes)

    n_val = int(round(len(scenes) * val_fraction))
    if val_fraction > 0:
        n_val = max(1, min(n_val, len(scenes) - 1)) if len(scenes) > 1 else 0
    val_scenes = set(scenes[:n_val])

    train = [r for r in records if str(r["scene"]) not in val_scenes]
    val = [r for r in records if str(r["scene"]) in val_scenes]
    return train, val


def check_no_scene_leak(train_records: Sequence[dict], val_records: Sequence[dict]) -> None:
    """守卫：训练集与验证集不许共享任何场景。"""
    train_scenes = {str(r["scene"]) for r in train_records}
    val_scenes = {str(r["scene"]) for r in val_records}
    overlap = train_scenes & val_scenes
    if overlap:
        shown = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            f"训练集与验证集共享 {len(overlap)} 个场景（如 {shown}）。\n"
            "这会造成数据泄漏、验证 PSNR 虚高。请用 data.split_by_scene 重新划分。"
        )


# --------------------------------------------------------------------------- #
# 增强
# --------------------------------------------------------------------------- #
def _augment_pair(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """对两张图施加**同一组**几何变换。

    刻意只做几何变换 —— 色彩抖动会改变噪声统计，对降噪任务是负作用。

    ⚠️ 配对数据必须严格同步：两次独立调用 `rng` 会让 GT 和 noisy 变换不一致，
    错一个像素这组数据就废了。
    """
    if rng.random() < 0.5:
        a, b = a[::-1], b[::-1]
    if rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    k = int(rng.integers(0, 4))
    if k:
        a, b = np.rot90(a, k), np.rot90(b, k)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DenoiseDataset(Dataset):
    """在线合成噪声的降噪数据集。

    Args:
        records: 索引记录列表
        noise_params: 标定好的 K / σ_read
        isp: 两条支路共用的 ISP 实例
        patch: 训练 patch 边长
        train: True 随机裁剪+增强+随机 ISO；False 中心裁剪+固定噪声实现
        backend: "png" 或 "lmdb"
        lmdb_path: backend=lmdb 时必填
        iso_choices: 采样用的 ISO 候选；默认取三台相机的公共 ISO
        seed: 训练侧的随机种子
    """

    def __init__(
        self,
        records: Sequence[dict],
        noise_params: NoiseParams,
        isp: ISP,
        patch: int = 128,
        train: bool = True,
        backend: str = "png",
        lmdb_path: str | Path | None = None,
        iso_choices: Sequence[int] | None = None,
        seed: int = 0,
        max_dn: float = DEFAULT_MAX_DN,
        cross_camera: bool = False,
        cond_dropout: float = 0.0,
        noise_corr_lag1: float = 0.0,
        noise_cross_channel: float = 0.0,
    ):
        super().__init__()
        self.records = list(records)
        if not self.records:
            raise ValueError("records 为空")

        self.noise_params = noise_params
        self.isp = isp
        self.patch = int(patch)
        self.train = bool(train)
        self.backend = backend
        self.max_dn = float(max_dn)
        # 交叉合成：用一台相机的噪声参数去噪另一台相机的干净图。
        # 只在训练生效 —— 验证集必须确定，随机换相机会让 PSNR 不可比。
        self.cross_camera = bool(cross_camera)
        # 条件 dropout：按概率把噪声参数换成"未知"标记，走盲降噪那条路。
        # 三机型方案下默认 0.0（我们总能查到准确参数），留着是为了消融。
        self.cond_dropout = float(cond_dropout)
        if not 0.0 <= self.cond_dropout <= 1.0:
            raise ValueError(f"cond_dropout 应在 [0,1]，收到 {cond_dropout}")

        # 合成噪声的真实性修正 —— 见 data/noise_model.py 顶部。
        # 只按 K/σ 加逐像素高斯噪声，和真实高感噪点差很远（实测 lag-1 自相关
        # 0.000 vs 真实 0.547），模型因此学不会去真实噪声。
        self.noise_corr_lag1 = float(noise_corr_lag1)
        self.noise_cross_channel = float(noise_cross_channel)

        if iso_choices:
            self.iso_choices = [int(i) for i in iso_choices]
        else:
            self.iso_choices = noise_params.common_isos() or noise_params.all_isos()
        if not self.iso_choices:
            raise ValueError("iso_choices 为空，检查 noise_params.json")

        self._check_camera_consistency()

        self.rng = np.random.default_rng(seed)

        self._lmdb_env = None
        self._txn = None
        if backend == "lmdb":
            if lmdb_path is None:
                raise ValueError("backend='lmdb' 时必须提供 lmdb_path")
            import lmdb  # 延迟导入：只用 png 时不强制装 lmdb

            self._lmdb_env = lmdb.open(
                str(lmdb_path), readonly=True, lock=False,
                readahead=False, max_readers=256,
            )
            self._txn = self._lmdb_env.begin(write=False)
        elif backend != "png":
            raise ValueError(f"未知 backend: {backend!r}（只支持 'png' / 'lmdb'）")

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def _load_source(self, src: str, h=None, w=None) -> np.ndarray:
        """读一张【线性 RGB】图，返回 (H,W,3) float32 [0,1]。"""
        if self.backend == "lmdb":
            raw = self._txn.get(str(src).encode("utf-8"))
            if raw is None:
                raise KeyError(f"LMDB 里没有 key {src!r}")
            if h is None or w is None:
                raise ValueError(f"LMDB 记录缺少 h/w：{src!r}")
            arr = np.frombuffer(raw, dtype=np.uint16)
            if arr.size != int(h) * int(w) * 3:
                raise ValueError(f"key {src!r} 数据量 {arr.size} 与 {h}x{w}x3 不符")
            return arr.reshape(int(h), int(w), 3).astype(np.float32) / 65535.0

        # 索引里存的是【相对项目根】的路径，这样整个项目目录可以整体搬走。
        # 相对路径按 PROJECT_ROOT 解析，不受当前工作目录影响。
        path = Path(str(src))
        if not path.is_absolute():
            from config import PROJECT_ROOT

            path = PROJECT_ROOT / path
        # PNG / TIFF 的统一读法在 data/imageio.py。
        # 通道顺序（cv2 读出来是 BGR）和位深（8/16）都在那里处理，
        # 散在各处写一定会有一处写错，而且错了看不出来。
        return read_linear(path)

    def _load_linear(self, rec: dict) -> np.ndarray:
        return self._load_source(rec["src"], rec.get("h"), rec.get("w"))

    def _crop_params(self, h: int, w: int, rng) -> Tuple[int, int]:
        p = self.patch
        if h < p or w < p:
            raise ValueError(
                f"图只有 {h}x{w}，裁不出 {p}x{p} 的 patch。"
                "把 dataset.patch 调小，或换更大的源图。"
            )
        if self.train:
            return int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))
        return (h - p) // 2, (w - p) // 2

    def _check_camera_consistency(self) -> None:
        """索引里的相机下标，必须和当前 noise_params.json 的相机顺序对得上。

        为什么要有这一条：`cam` 是 **build 时**按当时的 cameras 列表算出来的下标。
        标定文件后来一变（最常见的就是某台相机平场不合格、被 calibrate 自动剔除），
        这些下标就全错位了。后果分两种，两种都很糟：

        * 越界 → 一句 `IndexError: list index out of range`，完全看不出是配置问题；
        * 不越界 → **静默**把 A 相机的噪声加到 B 相机的干净图上，
          训练照跑、loss 照降、指标照出，但物理意义是错的。

        记录里同时存了相机**名字**，所以这里能反查，把静默错误变成响亮的报错。
        """
        names = self.noise_params.cameras
        bad = []
        for r in self.records:
            cam = int(r.get("cam", -1))
            name = r.get("camera")
            if not 0 <= cam < len(names) or (name is not None and names[cam] != name):
                bad.append((r.get("src"), cam, name))
        if bad:
            src, cam, name = bad[0]
            raise ValueError(
                f"索引里的相机下标和 noise_params.json 对不上了"
                f"（{len(bad)}/{len(self.records)} 条）。\n"
                f"  例如 {src}\n"
                f"       索引写的是 cam={cam}（{name}），"
                f"但 noise_params.json 的 cameras = {names}\n"
                f"  索引是在 `prepare_data.py build` 时按**当时**的标定文件顺序生成的。\n"
                f"  标定文件变了（例如某台相机被剔除）之后，必须重建索引：\n"
                f"      python data/prepare_data.py build --raw-dir shots\n"
                f"  否则轻则报错，重则把 A 相机的噪声静默加到 B 相机的图上。"
            )

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index]

        # 验证集必须【确定性】：同一个 index 每次拿到同样的裁剪、同样的 ISO、
        # 同样的噪声实现。否则跨实验的 PSNR 没法比，等于没做评测。
        rng = self.rng if self.train else np.random.default_rng(1234 + index)

        gt_linear = self._load_linear(rec)
        h, w = gt_linear.shape[:2]
        y, x = self._crop_params(h, w, rng)

        cam = int(rec["cam"])

        # 真实配对记录（来自 prepare_data build 的 pair 类别）：
        # 直接读拍出来的噪声图，不再在线合成。
        real_noisy = rec.get("noisy_src")
        if real_noisy:
            noisy_linear = self._load_source(real_noisy, rec.get("noisy_h"), rec.get("noisy_w"))
            if noisy_linear.shape[:2] != (h, w):
                raise ValueError(
                    f"{rec['scene']}: GT {h}x{w} 与 noisy "
                    f"{noisy_linear.shape[:2]} 尺寸不一致"
                )
            iso = int(rec.get("iso", self.iso_choices[0]))
            K, sigma_read = self.noise_params.get(cam, iso)
            # ★ 同一组裁剪参数作用到两张图上 —— 配对数据一旦错位就废了
            gt_linear = gt_linear[y:y + self.patch, x:x + self.patch]
            noisy_linear = noisy_linear[y:y + self.patch, x:x + self.patch]
        else:
            gt_linear = gt_linear[y:y + self.patch, x:x + self.patch]
            if self.train:
                iso = int(rng.choice(self.iso_choices))
            else:
                # 验证集按 index 轮转 ISO，保证每一档都被评到
                iso = int(self.iso_choices[index % len(self.iso_choices)])

            # ★ 噪声参数可以取自【另一台】相机。
            #   物理上完全合法（"给这张干净图加上 B 相机的噪声"），而且：
            #     * 把有效训练数据放大 N 倍（N = 3）
            #     * 逼模型学"噪声只跟噪声参数有关，跟哪台相机拍的无关"
            #       —— 三台相机之间的泛化就来自这里
            #   只在训练时启用；验证集必须确定，不能随机换相机。
            noise_cam = cam
            if self.cross_camera and self.train and self.noise_params.n_cameras > 1:
                noise_cam = int(rng.integers(0, self.noise_params.n_cameras))
                iso = int(rng.choice(self.noise_params.iso_list(noise_cam)))

            K, sigma_read = self.noise_params.get(noise_cam, iso)
            # wb = 该相机解码时的白平衡增益（索引里带的）。
            # K/σ_read 标定于白平衡**之前**，而 gt_linear 是**之后**的 ——
            # 不把增益传下去，R/B 通道的噪声会明显偏小。
            noisy_linear = add_raw_noise(
                gt_linear, K, sigma_read, rng, self.max_dn,
                wb_gains=rec.get("wb"),
                corr_lag1=self.noise_corr_lag1,
                cross_channel=self.noise_cross_channel,
            )

        gt_linear = np.ascontiguousarray(gt_linear)
        noisy_linear = np.ascontiguousarray(noisy_linear)

        # 几何增强必须用【同一组】参数同时作用到两张图
        if self.train:
            gt_linear, noisy_linear = _augment_pair(gt_linear, noisy_linear, rng)

        # ⚠️ 两条支路必须走【同一个】 ISP 实例、同一套参数（规划文档坑 #3）
        gt = self.isp.apply_np(gt_linear)
        noisy = self.isp.apply_np(noisy_linear)

        # 条件 dropout：把噪声参数换成"未知"标记（K = -1）。
        # 模型于是同时学会两条路 —— 知道噪声水平时更准，不知道时自己判断。
        # **只在训练时丢**：验证集必须确定，否则 PSNR 跨实验不可比。
        dropped = (
            self.train and self.cond_dropout > 0.0 and rng.random() < self.cond_dropout
        )

        to_chw = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return {
            "noisy": to_chw(noisy),
            "gt": to_chw(gt),
            # ★ 模型的唯一条件输入。K < 0 表示"不知道噪声水平"。
            "noise_k": torch.tensor(-1.0 if dropped else K, dtype=torch.float32),
            "noise_sigma": torch.tensor(
                -1.0 if dropped else float(sigma_read), dtype=torch.float32
            ),
            "cond_known": torch.tensor(0.0 if dropped else 1.0, dtype=torch.float32),
            # 以下三项**不进模型**，只用于报告、分组统计和排查
            "cam": torch.tensor(cam, dtype=torch.long),
            "iso": torch.tensor(iso, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def __del__(self):  # pragma: no cover - 进程退出时清理
        # 用 getattr 而不是 self._lmdb_env：__init__ 中途抛异常时（比如相机下标
        # 对不上），这个属性根本还没建出来，直接访问会在**真正的报错后面**再刷一段
        # AttributeError，看起来像坏了两处，把原因淹掉。
        if getattr(self, "_lmdb_env", None) is not None:
            try:
                self._lmdb_env.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_isp(isp_cfg: dict | None) -> ISP:
    """从配置构造 ISP。color_matrix 支持 "identity" 或 3x3 嵌套列表。"""
    isp_cfg = isp_cfg or {}
    m = isp_cfg.get("color_matrix", "identity")
    if isinstance(m, str):
        if m != "identity":
            raise ValueError(
                f"color_matrix 只支持 'identity' 或 3x3 列表，收到 {m!r}"
            )
        matrix = identity_matrix()
    else:
        matrix = np.asarray(m, dtype=np.float32)
    return ISP(ISPParams(color_matrix=matrix, gamma=float(isp_cfg.get("gamma", 2.2))))


def build_datasets(cfg: dict, seed: int = 0):
    """按配置构造 (train_ds, val_ds, noise_params)。"""
    from config import cfg_get  # 项目根在 sys.path 上（见 train.py 的 bootstrap）

    dcfg = cfg.get("data", cfg)
    noise_params = NoiseParams.load(cfg_get(dcfg, "noise_params", "noise_params.json"))
    isp = build_isp(dcfg.get("isp"))

    train_recs = load_index(cfg_get(dcfg, "train_index", "data/dummy/train.jsonl"))
    val_recs = load_index(cfg_get(dcfg, "val_index", "data/dummy/val.jsonl"))
    check_no_scene_leak(train_recs, val_recs)

    backend = cfg_get(dcfg, "backend", "png")
    lmdb_path = dcfg.get("lmdb_path")
    # 验证集指向**另一个** LMDB。train / val 是分开打包的
    # （按场景划分，同一场景不能跨两边），没有理由共用一个库。
    # 不填就退回 lmdb_path —— 兼容只建了一个库的老配置。
    val_lmdb_path = dcfg.get("val_lmdb_path") or lmdb_path
    patch = int(cfg_get(dcfg, "patch", 128))

    # 合成噪声的真实性修正 —— 数值由实测真实 RAW 噪声得到，
    # 详见 data/noise_model.py 顶部。不填就是逐像素独立的旧行为。
    nm = dcfg.get("noise_model") or {}

    common = dict(
        noise_params=noise_params,
        isp=isp,
        patch=patch,
        backend=backend,
        iso_choices=dcfg.get("iso_choices"),
        max_dn=float(cfg_get(dcfg, "max_dn", DEFAULT_MAX_DN)),
        cross_camera=bool(cfg_get(dcfg, "cross_camera_noise", False)),
        cond_dropout=float(cfg_get(dcfg, "cond_dropout", 0.0)),
        noise_corr_lag1=float(nm.get("corr_lag1") or 0.0),
        noise_cross_channel=float(nm.get("cross_channel") or 0.0),
    )
    train_ds = DenoiseDataset(train_recs, train=True, seed=seed,
                              lmdb_path=lmdb_path, **common)
    val_ds = DenoiseDataset(val_recs, train=False, seed=seed + 1,
                            lmdb_path=val_lmdb_path, **common)
    return train_ds, val_ds, noise_params


def build_loaders(cfg: dict, seed: int = 0):
    """构造 (train_loader, val_loader, noise_params)。"""
    from config import cfg_get

    train_ds, val_ds, noise_params = build_datasets(cfg, seed)
    dcfg = cfg.get("data", cfg)
    workers = int(cfg_get(dcfg, "num_workers", 4))
    batch = int(cfg_get(cfg.get("train", cfg), "batch", 4))

    # 没有 GPU 时 pin_memory 会刷警告且毫无收益，按设备决定
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch // 2),
        shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=pin,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader, noise_params
