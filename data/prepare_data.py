"""RAW → 线性 sRGB 数据管线。**全项目的关键路径。**

三个子命令，都从同一份 RAW 读起：

    # 1) 解码入库（B 的 D3 任务）
    python data/prepare_data.py build --raw-dir D:/shots --out data/real

    # 2) 噪声标定（A 的 D8-D9 任务）→ 产出 noise_params.json
    python data/prepare_data.py calibrate --raw-dir D:/shots --out noise_params.json

    # 3) 验收自检（拍完当天 + 入库后各跑一次）
    python data/prepare_data.py verify --raw-dir D:/shots

--------------------------------------------------------------------------
为什么这些设置一个都不能错
--------------------------------------------------------------------------

**① 关闭所有降噪。**
相机直出的 JPEG 已经做过机内降噪，噪声不是原始传感器噪声。
自己解码的话，如果开了 FBDD / 中值滤波，等于把要学的东西提前抹掉了。

**② gamma=(1,1) —— 输出线性，不是普通图片。**
因为噪声必须在线性域合成（物理模型只在线性域成立）。
**副作用：这些图用 Photoshop 打开会又暗又灰，那是正确的。**
想在 PS 里看，用 `data.imageio.write_srgb_preview()` 导预览。

**③ 同一场景的 GT 和 noisy 必须用同一组白平衡系数。**
相机会给不同 ISO 写入不同的 as-shot WB。各自按自己的 WB 解码，
两张图色调就不一致了 —— PSNR 会虚低十几个 dB，而且怎么调模型都救不回来。

本脚本的默认策略是 `--wb-mode fixed-per-camera`：
**从每台相机的第一张普通照片里读一次相机自己的 WB，然后全程固定用它。**
这样既保证了一致性，又不用我们凭空编造系数。

--------------------------------------------------------------------------
文件名规范
--------------------------------------------------------------------------

**推荐：文件夹模式 —— RAW 文件名一个字都不用改。**

    {raw_dir}/{相机}/{类别}/相机直出的文件名

    shots/Z6II/dark/DSC0100.NEF
    shots/Z6II/flat/DSC0300.NEF
    shots/5d3/dark/8M4B3875.CR2

* **类别**（dark / flat / clean / pair / demo）→ 由**文件夹名**决定
* **相机** → 从 **EXIF** 的 `Model` 字段读（`ILCE-7M3` → SonyA7M3）
* **ISO / 快门** → 从 **EXIF** 读
* 相机名认不出来时，退回用相机文件夹名（`5d3` → Canon5D3）

**兼容：文件名模式（旧）**，信息全写在文件名里：

    Z6II_dark_iso6400_1-60_07.NEF
    Z6II_flat_iso1600_ev03_10.NEF

两种模式可以混用，扫描时会自动识别。

⚠️ **空目录和「XX - 副本」不会被读到**（没有文件就进不来），
   但**有文件的副本目录会被当成第三台相机** —— 拍完记得清掉，
   否则重复帧会让测出来的 σ_read 偏低。

依赖：文件夹模式需要 `pip install exifread`。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402
from data.dataset import split_by_scene, write_index  # noqa: E402
from data.imageio import write_linear  # noqa: E402
from data.noise_model import NoiseParams  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".raw",
}

CATEGORIES = ("dark", "flat", "clean", "pair", "demo")

# 文件名里的简称 → noise_params.json 里的规范名
CAMERA_ALIASES = {
    "5d3": "Canon5D3", "5dmark3": "Canon5D3", "canon5d3": "Canon5D3",
    "eos5d3": "Canon5D3", "canon": "Canon5D3",
    "z6ii": "NikonZ6II", "z62": "NikonZ6II", "z6_2": "NikonZ6II",
    "nikonz6ii": "NikonZ6II", "nikon": "NikonZ6II",
    "a7m3": "SonyA7M3", "a7iii": "SonyA7M3", "a73": "SonyA7M3",
    "ilce7m3": "SonyA7M3", "sony": "SonyA7M3",
}

# 文件名：{相机}_{类别}_{场景}_{ISO}_{快门或档位}_{序号}
#
# 场景字段对 dark / flat 可以省略（它们本来就没有场景编号）：
#     Z6II_clean_s045_iso100_1-250_01.NEF    ← 有场景 s045
#     Z6II_pair_s012_iso6400_1-250_01.NEF
#     Z6II_dark_iso6400_1-60_07.NEF          ← 无场景
#     Z6II_flat_iso1600_ev03_10.NEF          ← 无场景，末段 ev03 是曝光档位
#
# `(?!iso)` 是必需的：场景名允许字母+数字，而 "iso6400" 恰好也符合这个形状，
# 不加负向前瞻的话正则会先把 "iso6400_" 当成场景吞掉。
NAME_RE = re.compile(
    r"^(?P<camera>[A-Za-z0-9]+)_"
    r"(?P<category>" + "|".join(CATEGORIES) + r")_"
    r"(?:(?P<scene>(?!iso)[A-Za-z0-9]+)_)?"
    r"iso(?P<iso>\d+)_"
    r"(?P<tail>[A-Za-z0-9\-]+)_"
    r"(?P<idx>\d+)$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# 文件名解析
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Shot:
    path: Path
    camera: str        # 规范名，如 NikonZ6II
    category: str      # dark | flat | clean | pair | demo
    scene: str         # 场景编号；flat 用 ev 档位
    iso: int
    shutter: str       # 文件名里的原始写法，如 "1-250"
    idx: int
    # 文件修改时间（≈拍摄时间）。只有平铺在 pair/ 下、没有场景子目录时才用得上，
    # 见 autogroup_flat_pairs()。给默认值是为了兼容别处的构造。
    mtime: float = 0.0

    @property
    def scene_key(self) -> str:
        return f"{self.camera}_{self.scene}"

    @property
    def shutter_seconds(self) -> float:
        return parse_shutter(self.shutter)

    def out_name(self, ext: str) -> str:
        """解码后的输出文件名：保留原始文件名主干，只换扩展名。

        ⚠️ **不要用规范化后的相机名重建文件名。**
        原始文件叫 `Z6II_...`（简称），规范化后是 `NikonZ6II`。
        拿规范化名重建出来的路径和磁盘上的对不上，结果是**每一个文件都
        "找不到"**，而且不会报错，只会静默地一条索引都建不出来。
        """
        return self.path.stem + ext


def normalize_camera(raw: str) -> str:
    key = raw.lower().replace("-", "").replace("_", "")
    if key in CAMERA_ALIASES:
        return CAMERA_ALIASES[key]
    # 已经是规范名就直接用
    for canon in ("Canon5D3", "NikonZ6II", "SonyA7M3"):
        if key == canon.lower():
            return canon
    raise ValueError(
        f"认不出相机 {raw!r}。可用简称：{sorted(set(CAMERA_ALIASES))}，"
        "或在 CAMERA_ALIASES 里补一条"
    )


def parse_shutter(s: str) -> float:
    """'1-250' → 1/250；'1' → 1；'2-1' → 2。"""
    if "-" in s:
        num, den = s.split("-", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _safe_mtime(path: Path) -> float:
    """文件的修改时间；文件不存在就返回 0。

    ⚠️ 必须有这个 try —— `parse_name` 是**纯字符串解析**，自检里拿
    "Z6II_dark_iso6400_1-60_07.NEF" 这种假文件名直接调它，磁盘上并没有
    这个文件。直接 stat() 会抛 FileNotFoundError，把自检的「RAW 管线」
    那一项打成 FAIL。
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def parse_name(path: Path) -> Optional[Shot]:
    """解析文件名。不符合规范的返回 None（会被计入 skipped）。"""
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    try:
        camera = normalize_camera(m.group("camera"))
    except ValueError:
        return None

    category = m.group("category").lower()
    scene = m.group("scene")
    tail = m.group("tail")

    if category == "flat":
        # 平场没有场景，末段 ev03 是曝光档位 —— 标定要按档位分组拟合 PTC，
        # 所以把它当作 scene 用。
        scene = tail
    elif scene is None:
        # 暗帧：没有场景，给个固定值占位
        scene = category

    return Shot(
        path=path,
        camera=camera,
        category=category,
        scene=scene,
        iso=int(m.group("iso")),
        shutter=tail,
        idx=int(m.group("idx")),
        mtime=_safe_mtime(path),
    )


# --------------------------------------------------------------------------- #
# EXIF 读取（文件夹模式靠它拿相机 / ISO / 快门）
# --------------------------------------------------------------------------- #
# EXIF 里的机型字符串 → 项目内的相机名。
# 各家写法差别很大（尼康 "NIKON Z 6_2"、索尼 "ILCE-7M3"），只能靠关键词匹配。
EXIF_CAMERA_HINTS = [
    (("5d mark iii", "eos 5d mark iii", "5d3", "5dmark3"), "Canon5D3"),
    (("z 6_2", "z6_2", "z 6ii", "z6ii", "z 6 ii"), "NikonZ6II"),
    (("ilce-7m3", "a7 iii", "a7iii", "a7m3"), "SonyA7M3"),
]


def camera_from_model(model: str) -> Optional[str]:
    """EXIF 机型字符串 → 规范相机名。认不出来返回 None。"""
    low = str(model).lower()
    for keys, name in EXIF_CAMERA_HINTS:
        if any(k in low for k in keys):
            return name
    return None


def read_exif(path: Path) -> Tuple[str, int, str]:
    """读 (机型, ISO, 快门字符串)。读不到的部分返回空值 / 0。

    快门会转成文件名安全的写法（`1/60` → `1-60`）。
    """
    try:
        import exifread
    except ImportError as exc:
        raise ImportError(
            "文件夹模式需要 exifread 来读 ISO / 机型：\n"
            "    pip install exifread"
        ) from exc

    model, iso, shutter = "", 0, ""
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(
                f, details=False, stop_tag="EXIF ExposureTime"
            )
    except Exception:  # noqa: BLE001 - 个别文件坏了不该中断整批扫描
        return model, iso, shutter

    model = str(tags.get("Image Model", "")).strip()

    raw_iso = tags.get("EXIF ISOSpeedRatings")
    if raw_iso is not None:
        try:
            # exifread 可能返回 "100" 或 "[100, 200]"
            iso = int(str(raw_iso).strip("[] ").split(",")[0])
        except (ValueError, IndexError):
            iso = 0

    raw_exp = tags.get("EXIF ExposureTime")
    if raw_exp is not None:
        shutter = str(raw_exp).strip().replace("/", "-").replace(" ", "")

    return model, iso, shutter


def _trailing_int(stem: str) -> int:
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def shot_from_folders(path: Path, root: Path) -> Optional[Shot]:
    """从 `{相机}/{类别}/[场景/]文件` 结构造一个 Shot。

    相机优先读 EXIF（可靠），读不到才退回文件夹名。
    ISO / 快门**只从 EXIF 读** —— 文件名里没有这些信息。

    场景键（`scene`）按类别分别处理，**这个区分很关键**：

    ============  ==========================================================
    dark / flat   必须留空。标定要按曝光档分组，留空后上层会退回用快门当键。
                  填了文件名反而会把同一个曝光档拆散。
    clean / demo  用文件名。一张照片本身就是一个独立场景（每张各自合成
                  噪声），不这么做的话 100 多张图会全被当成**同一个**场景，
                  train/val 根本切不开，验证集直接是空的。
    pair          取自 `pair/` 下面的**场景子目录**。同一场景的 GT 和 noisy
                  必须归到同一组，所以这一级目录不能省。
    ============  ==========================================================
    """
    rel = path.relative_to(root).parts
    cat_idx = next((i for i, part in enumerate(rel[:-1])
                    if part.lower() in CATEGORIES), None)
    if cat_idx is None or cat_idx == 0:
        return None
    cam_dir = rel[cat_idx - 1]
    cat_dir = rel[cat_idx].lower()
    scene_dirs = rel[cat_idx + 1:-1]        # 类别与文件之间夹着的目录 = 场景

    model, iso, shutter = read_exif(path)

    camera = camera_from_model(model) if model else None
    if camera is None:
        try:
            camera = normalize_camera(cam_dir)
        except ValueError:
            return None

    if iso <= 0:
        # EXIF 里没有 ISO 的文件用不了 —— 标定和合成都需要它
        return None

    if scene_dirs:
        scene = "/".join(scene_dirs)
    elif cat_dir in ("clean", "demo"):
        scene = path.stem
    else:
        scene = ""

    return Shot(
        path=path,
        camera=camera,
        category=cat_dir,
        scene=scene,
        iso=iso,
        shutter=shutter or "na",
        idx=_trailing_int(path.stem),
        mtime=_safe_mtime(path),
    )


def scan_raws(raw_dir: Path) -> Tuple[List[Shot], List[Path]]:
    """扫描目录，返回 (解析成功的 shot 列表, 无法解析的文件列表)。

    **优先用文件夹结构**（推荐，文件名一个字都不用改）::

        {raw_dir}/{相机}/{类别}/xxxx.CR2
        例：shots/Z6II/dark/DSC0100.NEF

    `pair` 类别在类别和文件之间最好多一级「场景目录」::

        shots/A7M3/pair/s001/gt_iso200.ARW
        shots/A7M3/pair/s001/noisy_iso51200.ARW

    同一场景的 GT 和 noisy 放进同一个场景目录 —— 这是配对**最可靠**的依据。

    **平铺也行**：直接丢在 ``shots/A7M3/pair/`` 下、用相机原生文件名
    （``8M4B4986.CR2`` 这种）也可以，``build`` 会按拍摄时间自动补场景编号，
    见 `autogroup_flat_pairs()`。但那是推断，跑完要核对打印出来的分组。
    clean / demo / dark / flat 都不需要这一级。

    认不出来才回退到旧的文件名解析::

        Z6II_pair_s012_iso6400_1-250_01.NEF

    ⚠️ 文件夹模式下空目录和「XX - 副本」这种都会被跳过（没有文件就不会进来），
       但**有文件的副本目录会被当成第三台相机**，拍完记得清掉。
    """
    raw_dir = Path(raw_dir)
    shots: List[Shot] = []
    bad: List[Path] = []
    n_folder = n_name = 0

    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in RAW_EXTS:
            continue

        s = shot_from_folders(p, raw_dir)
        if s is not None:
            n_folder += 1
        else:
            s = parse_name(p)
            if s is not None:
                n_name += 1

        if s is None:
            bad.append(p)
        else:
            shots.append(s)

    if shots and n_folder:
        print(f"  [扫描] 文件夹模式 {n_folder} 个"
              f"{f'，文件名模式 {n_name} 个' if n_name else ''}")
    return shots, bad


# --------------------------------------------------------------------------- #
# 平铺 pair 的自动分组
# --------------------------------------------------------------------------- #
def autogroup_flat_pairs(
    shots: Sequence[Shot], gap_s: float
) -> Tuple[List[Shot], List[str], List[str]]:
    """给**没有场景子目录**的 pair 图按拍摄时间补一个场景编号。

    正常做法是拍完自己建目录：``shots/A7M3/pair/s001/{gt,noisy}.ARW``。
    但相机导出来的常常就是平铺的一堆 ``8M4B4986.CR2`` —— 原生文件名里
    没有任何场景信息，手工分一百多个文件也不现实。

    这里靠**拍摄时间**自动分：同一个场景的几张是连着拍的（构图不动、只改
    ISO），换场景要重新构图对焦，中间必然隔得更久。

    ⚠️ 这是**推断**，不是事实。前提是文件的 mtime 还是拍摄时间（从存储卡
    直接拷出来通常是；经网盘/聊天工具中转过的就不是了）。所以调用方**必须**
    把分组结果打印出来给人核对，见 cmd_build。

    返回 ``(新的 shots, 报告行, 问题行)``。报告行缩进两级，便于直接 print。
    """
    flat = [s for s in shots if s.category == "pair" and not s.scene]
    if not flat:
        return list(shots), [], []

    by_cam: Dict[str, List[Shot]] = {}
    for s in flat:
        by_cam.setdefault(s.camera, []).append(s)

    fixed: Dict[Path, Shot] = {}
    report: List[str] = []
    problems: List[str] = []

    for cam, group in sorted(by_cam.items()):
        group.sort(key=lambda s: (s.mtime, s.path.name))
        if len(group) < 2:
            problems.append(f"{cam}: 只有 {len(group)} 张 pair，凑不成对")
            continue

        # 相邻两张隔得够久 → 换场景了
        chunks: List[List[Shot]] = []
        cur = [group[0]]
        for a, b in zip(group, group[1:]):
            if b.mtime - a.mtime > gap_s:
                chunks.append(cur)
                cur = [b]
            else:
                cur.append(b)
        chunks.append(cur)

        t0 = time.strftime("%H:%M:%S", time.localtime(group[0].mtime))
        t1 = time.strftime("%H:%M:%S", time.localtime(group[-1].mtime))
        report.append(f"  {cam}: {len(group)} 张 → {len(chunks)} 组（{t0} ~ {t1}）")

        for gi, g in enumerate(chunks, 1):
            scene = f"auto{gi:03d}"
            span = g[-1].mtime - g[0].mtime
            detail = "、".join(f"ISO{s.iso}" for s in g)
            report.append(f"    {scene}  {len(g)} 张  跨 {span:.0f}s  {detail}")
            if len(g) < 2:
                problems.append(
                    f"{cam}/{scene} 只有 1 张（ISO{g[0].iso}，{g[0].path.name}）—— 配不成对"
                )
            for s in g:
                fixed[s.path] = replace(s, scene=scene)

    out = [fixed.get(s.path, s) for s in shots]
    return out, report, problems


# --------------------------------------------------------------------------- #
# 白平衡
# --------------------------------------------------------------------------- #
def read_camera_wb(path: Path) -> Tuple[float, float, float, float]:
    """读相机写进 RAW 的 as-shot 白平衡系数（4 个，对应 CFA 通道）。"""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        wb = raw.camera_whitebalance
    if wb is None or len(wb) != 4 or min(wb) <= 0:
        raise ValueError(f"{path.name}: 读不到有效的白平衡系数")
    return tuple(float(v) for v in wb)


def resolve_white_balance(
    shots: Sequence[Shot],
    mode: str,
    explicit: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    """给每台相机定一组白平衡，全程固定使用。

    模式：
        fixed-per-camera（默认）—— 从每台相机第一张**普通照片**里读一次 WB，
                                   然后全程固定。既保证一致性，又不用编造系数。
        explicit             —— 用 --wb 显式指定
        as-shot              —— 每张各用各的（**会破坏一致性，除非你很清楚在做什么**）
    """
    cameras = sorted({s.camera for s in shots})
    if mode == "explicit":
        if not explicit:
            raise ValueError("--wb-mode explicit 需要同时用 --wb 指定系数")
        out = {}
        for cam in cameras:
            if cam not in explicit:
                raise ValueError(f"--wb 里缺少 {cam}")
            out[cam] = tuple(float(v) for v in explicit[cam])
        return out

    if mode == "as-shot":
        print("⚠️  --wb-mode as-shot：每张图各用各的白平衡。")
        print("⚠️  这会让 GT 和 noisy 的色调不一致，PSNR 会虚低。除非明确知道后果，别用。")
        return {cam: (0.0, 0.0, 0.0, 0.0) for cam in cameras}  # 0 表示"用文件自带的"

    # fixed-per-camera
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for cam in cameras:
        # ★ 从 clean / pair / demo 里挑，绝不能从 dark / flat 里挑 ——
        #   盖着镜头盖的暗帧算出来的自动白平衡是垃圾。
        candidates = [
            s for s in shots
            if s.camera == cam and s.category in ("clean", "pair", "demo")
        ]
        if not candidates:
            raise ValueError(
                f"{cam}: 找不到任何 clean/pair/demo 照片来提取白平衡。"
                "每台相机至少要有一张正常曝光的照片。"
            )
        # 取 ISO 最低的那张（最干净、最接近中性）
        ref = min(candidates, key=lambda s: (s.iso, str(s.path)))
        wb = read_camera_wb(ref.path)
        out[cam] = wb
        print(f"  [WB] {cam:<12} 取自 {ref.path.name}")
        print(f"       {[round(v, 3) for v in wb]}")
    return out


# --------------------------------------------------------------------------- #
# RAW 读取
# --------------------------------------------------------------------------- #
def postprocess_kwargs(wb: Sequence[float]) -> dict:
    """★ 解码参数。改这里之前先读模块开头那三条。"""
    import rawpy

    kw = dict(
        output_bps=16,
        output_color=rawpy.ColorSpace.sRGB,   # 色彩矩阵交给 rawpy
        gamma=(1, 1),                          # ★ 线性
        no_auto_bright=True,                   # ★ 关自动亮度
        use_camera_wb=False,                   # ★ 不用机内白平衡
        half_size=False,
        median_filter_passes=0,                # ★ 关中值滤波
        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,  # ★ 关 FBDD 降噪
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    )
    if wb and any(v > 0 for v in wb):
        kw["user_wb"] = [float(v) for v in wb]
    else:
        kw["use_camera_wb"] = True   # as-shot 模式
    return kw


def decode_linear(path: Path, wb: Sequence[float]) -> np.ndarray:
    """RAW → 线性 sRGB，float32 [0,1]，RGB 顺序。"""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(**postprocess_kwargs(wb))
    return rgb.astype(np.float32) / 65535.0


def read_raw_dn(path: Path, subtract_black: bool = True) -> np.ndarray:
    """读 RAW 的原始 DN 值（**不做任何 ISP**）。

    标定必须用这个，不能用 decode_linear —— 光子转移曲线要在
    线性、未去马赛克、未应用色彩矩阵的原始数据上做。

    ⚠️ 返回 float64 的 (H, W) 单通道平面：一个像素一个值。内存 = H×W×8 字节，
    22MP 的一帧就是 177MB —— 调用方**不要**把几十帧 stack 起来，见
    `_accumulate_time_stats`。
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        img = raw.raw_image_visible.astype(np.float64)
        if not subtract_black:
            return img

        pattern = np.asarray(raw.raw_pattern)
        levels = np.asarray(raw.black_level_per_channel, dtype=np.float64)
        # 原地减黑电平。之前是先造一份和整幅图一样大的 bl 再相减，
        # 白白多占一份平面（22MP → 177MB），直接改 img 的四个子栅格更省也更快。
        for dy in range(2):
            for dx in range(2):
                c = int(pattern[dy % pattern.shape[0], dx % pattern.shape[1]])
                img[dy::2, dx::2] -= levels[c]
    return img


# --------------------------------------------------------------------------- #
# 拜耳平面（给"拜耳域合成噪声"用，见 data/noise_model.py 顶部）
# --------------------------------------------------------------------------- #
# CFA 相位（按行展平的 2x2）→ OpenCV 的 Bayer 代码名。
# 0=R 1=G1 2=B 3=G2，和 rawpy 的约定一致。
#   (0,1,3,2) = [R G; G B] → RGGB
#   (1,0,2,3) = [G R; B G] → GRBG
#   (3,2,1,0) = [G B; R G] → GBRG
#   (2,3,0,1) = [B G; G R] → BGGR
BAYER_NAMES = {
    (0, 1, 3, 2): "RG",
    (1, 0, 2, 3): "GR",
    (3, 2, 1, 0): "GB",
    (2, 3, 0, 1): "BG",
}


def bayer_cv2_code(name: str, edge_aware: bool = False) -> int:
    """Bayer 名（"RG" 这种）→ cv2.cvtColor 的代码。

    跨通道那一组（`_EA`）理论上更接近真实相机的 AHD，但**实测对噪声的
    通道间相关性没有帮助** —— 相关主要来自去马赛克算法之外。
    """
    import cv2

    table = {
        "RG": (cv2.COLOR_BayerRG2RGB_EA, cv2.COLOR_BayerRG2RGB),
        "GR": (cv2.COLOR_BayerGR2RGB_EA, cv2.COLOR_BayerGR2RGB),
        "GB": (cv2.COLOR_BayerGB2RGB_EA, cv2.COLOR_BayerGB2RGB),
        "BG": (cv2.COLOR_BayerBG2RGB_EA, cv2.COLOR_BayerBG2RGB),
    }
    if name not in table:
        raise ValueError(f"不认识的 Bayer 名 {name!r}，可选 {sorted(table)}")
    ea, plain = table[name]
    return ea if edge_aware else plain


def read_mosaic(path: Path) -> Tuple[np.ndarray, str]:
    """读 RAW 的**原始拜耳平面**（减黑电平）→ (uint16 二维数组, Bayer 名)。

    和 `read_raw_dn` 的区别：这里返回 uint16（用来存盘），并且顺带把
    CFA 相位翻译成 cv2 认的名字。

    **为什么需要它**：噪声在**拜耳阵列**上就存在了。先加噪再去马赛克，
    空间相关和通道结构会**自然**正确；在已经去马赛克的 RGB 上加白噪声，
    就必须靠手工拟合参数去凑（见 data/noise_model.py 顶部）。
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        img = raw.raw_image_visible.astype(np.float64)
        pattern = np.asarray(raw.raw_pattern)
        levels = np.asarray(raw.black_level_per_channel, dtype=np.float64)
        for dy in range(2):
            for dx in range(2):
                img[dy::2, dx::2] -= levels[int(pattern[dy, dx])]

    name = BAYER_NAMES.get(tuple(int(v) for v in pattern.ravel()))
    if name is None:
        raise ValueError(f"{path.name}: 不认识的 CFA 相位 {pattern.tolist()}")
    return np.clip(img, 0.0, 65535.0).round().astype(np.uint16), name


def _mosaic_job(job: Tuple[str, str]) -> Tuple[str, str, str, str]:
    """解码一个 RAW 的拜耳平面并写成 16-bit 灰度 PNG。"""
    import cv2

    src, dst = job
    try:
        mosaic, name = read_mosaic(Path(src))
        if not cv2.imwrite(str(dst), mosaic):
            return ("fail", src, "", "cv2.imwrite 返回 False")
        return ("ok", src, name, "")
    except Exception as exc:  # noqa: BLE001
        return ("fail", src, "", f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 子命令：build
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 并行执行
# --------------------------------------------------------------------------- #
def init_multiprocessing() -> str:
    """设定多进程启动方式。**Linux 上必须用 spawn，否则会死锁。**

    原因：rawpy 内部用了 OpenMP，而 Linux 默认的 `fork` 启动方式会让
    「fork + OpenMP」死锁 —— 表现是 worker 进程被内核杀掉，主进程报
    `BrokenProcessPool: A process in the process pool was terminated abruptly`。

    官方说明：
    https://github.com/letmaik/rawpy#im-getting-deadlocks-when-using-multiprocessing-on-linux

    代价：spawn 会让每个 worker 重新 import 一遍模块（含 torch），启动慢几秒。
    一次性开销，可以接受。
    """
    import multiprocessing as mp

    current = mp.get_start_method(allow_none=True)
    if current == "spawn":
        return current
    try:
        mp.set_start_method("spawn", force=True)
        return "spawn"
    except RuntimeError:
        # 进程池已经启动过，改不了了 —— 至少告诉调用方当前是什么
        return current or "unknown"


class _NullPool:
    """`--workers 0` 时的替身：单进程直接跑，完全不建子进程。

    接口和 ProcessPoolExecutor 兼容（submit / map / with），
    所以调用处一行都不用改。调试时用它最省事。
    """

    def submit(self, func, *a, **kw):
        from concurrent.futures import Future

        fut: Future = Future()
        try:
            fut.set_result(func(*a, **kw))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut

    def map(self, func, jobs):
        return [func(j) for j in jobs]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_pool(workers: int):
    """workers <= 0 返回单进程替身；否则返回进程池。

    ⚠️ 调用前必须先执行过 `init_multiprocessing()`，否则 Linux 上 rawpy 会死锁。
    """
    if workers <= 0:
        return _NullPool()
    return ProcessPoolExecutor(max_workers=workers)


def _decode_job(job: Tuple[str, str, Sequence[float], int]) -> Tuple[str, str, int, str]:
    src, dst, wb, bits = job
    try:
        rgb = decode_linear(Path(src), wb)
        write_linear(Path(dst), rgb, bit_depth=bits)
        return ("ok", src, bits, "")
    except Exception as exc:  # noqa: BLE001
        return ("fail", src, bits, f"{type(exc).__name__}: {exc}")


def cmd_build(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"ERROR: 找不到目录 {raw_dir}")
        return 1

    shots, bad = scan_raws(raw_dir)
    if not shots:
        print(f"ERROR: {raw_dir} 里没有符合命名规范的 RAW")
        print("命名规范：{相机}_{类别}_{场景}_{ISO}_{快门}_{序号}.CR2/NEF/ARW")
        if bad:
            print(f"\n有 {len(bad)} 个文件解析失败，前几个：")
            for p in bad[:5]:
                print(f"  {p.name}")
        return 1

    by_cat: Dict[str, int] = {}
    for s in shots:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    print(f"扫描到 {len(shots)} 个 RAW：")
    for cat in CATEGORIES:
        if cat in by_cat:
            print(f"  {cat:<6} {by_cat[cat]:>5}")
    if bad:
        print(f"  [跳过] {len(bad)} 个文件名不符合规范")
        for p in bad[:3]:
            print(f"          例如 {p.name}")

    # ---- pair 分组 ----
    # 首选「场景子目录」：确定、无歧义。
    # 没有子目录就按**拍摄时间**自动分 —— 相机导出来的常常是平铺的原生文件名
    # （8M4B4986.CR2 这种），里面一个字场景信息都没有。
    # 自动分是**推断**，所以把结果整段打印出来给人核对，别闷头往下跑 ——
    # 分错了不会报错，只会把 A 场景的干净图和 B 场景的噪声图配成一对，
    # loss 照降、图是错的。
    flat_pairs = [s for s in shots if s.category == "pair" and not s.scene]
    if flat_pairs:
        print(f"\n⚠️  {len(flat_pairs)} 张 pair 平铺在 pair/ 下，没有场景子目录。")
        print(f"    按拍摄时间自动分组"
              f"（相邻两张间隔 > {args.pair_gap:.0f} 秒就算换场景）……")
        shots, report, problems = autogroup_flat_pairs(shots, args.pair_gap)

        n_groups = 0
        for line in report:
            print(line)
            if line.strip().startswith("auto"):
                n_groups += 1

        if n_groups == 0:
            print("\n❌ 一组都没分出来 —— pair 目录里没有能配对的文件。")
            return 1

        for line in problems:
            print(f"    ⚠️  {line}")

        if n_groups == 1 and len(flat_pairs) > 4:
            print(f"\n⚠️  全部 {len(flat_pairs)} 张被并成了 1 组，说明间隔阈值太大。")
            print("    把 --pair-gap 调小（比如 2）重跑。**别用这一版继续** ——")
            print("    一组里只有 ISO 最低那张当 GT，其余全被当成它的噪声图。")
        print("\n    ↑ 每一组应该是「一张低 ISO + 一张高 ISO」。"
              "组切得不对就调 --pair-gap 重跑。")

    out_dir = PROJECT_ROOT / args.out
    lin_dir = out_dir / "linear"

    # 只有 clean / pair / demo 需要解码成图片
    # dark 和 flat 是给 calibrate 用的，解码成图片没有意义（还占大量空间）
    to_decode = [s for s in shots if s.category in set(args.categories)]
    if not to_decode:
        print(f"\n没有需要解码的类别（--categories {','.join(args.categories)}）")
        return 1

    print("\n确定白平衡：")
    # ★ 只对**这次真的要解码**的相机求白平衡。
    #   这里原来传的是全部 shots，于是只要"某台相机没拍过 clean/pair/demo"，
    #   整条命令就直接失败 —— 哪怕那台相机这次根本不参与解码
    #   （比如只标定了、还没拍图）。白平衡只服务于解码，
    #   所以按参与解码的相机来求才是对的。
    wb_map = resolve_white_balance(to_decode, args.wb_mode, args.wb)

    lin_dir.mkdir(parents=True, exist_ok=True)

    # 输出名保留原始文件名主干（见 Shot.out_name）。同名冲突会让两个不同
    # 目录下的文件互相覆盖，必须先查出来。
    stems: Dict[str, List[Path]] = {}
    for s in to_decode:
        stems.setdefault(s.path.stem, []).append(s.path)
    collisions = {k: v for k, v in stems.items() if len(v) > 1}
    if collisions:
        print(f"\n⚠️  有 {len(collisions)} 组同名文件（会互相覆盖），请先改名：")
        for stem, paths in list(collisions.items())[:3]:
            print(f"   {stem}")
            for p in paths:
                print(f"     {p}")

    jobs = []
    for s in to_decode:
        dst = lin_dir / s.out_name(args.ext)
        if args.skip_existing and dst.exists():
            continue
        jobs.append((str(s.path), str(dst), wb_map[s.camera], 16))

    print(f"\n待解码 {len(jobs)} 张"
          f"{'（其余已存在，跳过）' if len(jobs) < len(to_decode) else ''}，"
          f"并行度 {args.workers}")

    done, failed = 0, []
    if jobs:
        with make_pool(args.workers) as pool:
            futures = [pool.submit(_decode_job, j) for j in jobs]
            for fut in as_completed(futures):
                status, src, _bits, err = fut.result()
                done += 1
                if status == "fail":
                    failed.append((src, err))
                if done % 20 == 0 or done == len(jobs):
                    print(f"  解码 {done}/{len(jobs)} ...", flush=True)

    if failed:
        print(f"\n⚠️  {len(failed)} 张解码失败：")
        for src, err in failed[:5]:
            print(f"   {Path(src).name}: {err}")

    # ---------------- 拜耳平面（可选）---------------- #
    # 给"拜耳域合成噪声"用：先加噪再去马赛克，空间/通道结构自然正确。
    # 见 data/noise_model.py 顶部。
    mosaic_dir = (out_dir / "mosaic") if args.save_mosaic else None
    bayer_by_cam: Dict[str, str] = {}
    if mosaic_dir is not None:
        mosaic_dir.mkdir(parents=True, exist_ok=True)

        # CFA 相位是**相机**属性，不随照片变 —— 每台读一张就够，
        # 这样即使图片全被 --skip-existing 跳过也能拿到相位。
        for cam in sorted({s.camera for s in to_decode}):
            s0 = next(s for s in to_decode if s.camera == cam)
            try:
                _, bayer_by_cam[cam] = read_mosaic(s0.path)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️  {cam}: 读不到 CFA 相位（{exc}），这台的拜耳平面会跳过")
        print(f"\n拜耳平面：CFA 相位 {bayer_by_cam}")

        mjobs = [
            (str(s.path), str(mosaic_dir / f"{s.path.stem}.png"))
            for s in to_decode
            if s.camera in bayer_by_cam
            and not (args.skip_existing and (mosaic_dir / f"{s.path.stem}.png").exists())
        ]
        print(f"待提取 {len(mjobs)} 张"
              f"{'（其余已存在，跳过）' if len(mjobs) < len(to_decode) else ''}")
        mfail = 0
        if mjobs:
            with make_pool(args.workers) as pool:
                for st, src, _name, err in pool.map(_mosaic_job, mjobs):
                    if st != "ok":
                        mfail += 1
                        if mfail <= 5:
                            print(f"   [失败] {Path(src).name}: {err}")
        print(f"  拜耳平面 {len(mjobs) - mfail}/{len(mjobs)} 张完成 → {mosaic_dir}")

    # ---------------- 建立索引 ---------------- #
    records = build_index(to_decode, lin_dir, out_dir, args.ext,
                          wb_map=wb_map,
                          mosaic_dir=mosaic_dir,
                          bayer_by_cam=bayer_by_cam)
    print(f"\n生成索引：{len(records)} 条")
    n_clean = sum(1 for r in records if "noisy_src" not in r)
    n_pair = len(records) - n_clean
    print(f"  合成用（干净图）{n_clean} 条")
    print(f"  真实配对        {n_pair} 条")

    train, val = split_by_scene(records, args.val_frac, args.seed)
    n_tr = write_index(out_dir / "train.jsonl", train)
    n_va = write_index(out_dir / "val.jsonl", val)

    print(f"  train.jsonl  {n_tr} 条")
    print(f"  val.jsonl    {n_va} 条  （按场景划分，不共享 scene）")
    print(f"\n输出目录 {out_dir}")

    if args.make_preview:
        n = write_previews(records, out_dir / "preview", args.preview_count)
        print(f"已导出 {n} 张预览图 → {out_dir / 'preview'}")
        print("（预览图套过 gamma，能在 Photoshop 里正常看；**不要拿去做训练**）")

    print("\n下一步：")
    print("  python data/prepare_data.py verify --raw-dir", args.raw_dir)
    print("  然后改 configs/base.yaml 的 data.train_index / val_index 指向新索引")
    return 0


def camera_index_map() -> Dict[str, int]:
    """相机名 → 模型输入用的整数索引。

    顺序必须与 `noise_params.json` 的 `cameras` 一致，否则模型学到的
    "相机 0" 和报告里写的对不上 —— 而条件嵌入是按这个索引查表的。
    """
    path = PROJECT_ROOT / "noise_params.json"
    if path.exists():
        try:
            return {c: i for i, c in enumerate(NoiseParams.load(path).cameras)}
        except Exception:  # noqa: BLE001 - 标定文件坏了不该阻断解码
            pass
    return {"Canon5D3": 0, "NikonZ6II": 1, "SonyA7M3": 2}


def wb_rgb_gains(wb: Sequence[float]) -> List[float]:
    """4 个 CFA 白平衡系数 → RGB 三通道的**噪声放大倍数**。

    相机写进 RAW 的是 [R, G1, B, G2] 四个值，去马赛克后两个绿通道合成一个 G。
    归一化到最小值为 1，就得到"这个通道被整体放大了多少倍"。

    **为什么噪声模型需要它**：白平衡是逐通道乘一个常数，信号和噪声一起放大。
    而 K / σ_read 是在白平衡**之前**的传感器 DN 上标定的。不把这个倍数告诉
    噪声模型，R/B 通道的噪声就会明显偏小 —— 实测索尼 A7M3 差到 1.56 倍，
    后果是模型遇到真实噪点根本不认识（见 data/noise_model.py 顶部）。

    例：索尼实测 [2664, 1024, 1772, 1024] → [2.6016, 1.0, 1.7305]
    """
    r, g1, b, g2 = (float(v) for v in wb)
    gains = [r, 0.5 * (g1 + g2), b]
    m = min(gains)
    if m <= 0:
        return [1.0, 1.0, 1.0]
    return [round(v / m, 4) for v in gains]


def build_index(
    shots: Sequence[Shot],
    lin_dir: Path,
    out_dir: Path,
    ext: str,
    wb_map: Optional[Mapping[str, Sequence[float]]] = None,
    mosaic_dir: Optional[Path] = None,
    bayer_by_cam: Optional[Mapping[str, str]] = None,
) -> List[dict]:
    """把 shot 列表变成训练索引。

    * `clean` / `demo` → 一条记录一张干净图，噪声由 dataset 在线合成
    * `pair` → **按场景配对**：ISO 最低的那张当 GT，其余每张噪声图各出一条记录，
      带 `noisy_src` 字段。`dataset.py` 看到这个字段就直接读拍出来的噪声图，
      不再合成 —— 这是阶段二（真实数据微调）的数据来源。

    `scene` 字段是 train/val 划分的唯一依据，**同一场景的 GT 和 noisy
    共用同一个值**。填错了就会数据泄漏，验证 PSNR 虚高好几个 dB。
    """
    cam_map = camera_index_map()
    unknown = sorted({s.camera for s in shots} - set(cam_map))
    if unknown:
        raise ValueError(
            f"这些相机不在 noise_params.json 的 cameras 列表里：{unknown}。"
            "先在 noise_params.json 里补上，或检查文件名里的相机简称。"
        )

    def out_path(s: Shot) -> Path:
        return lin_dir / s.out_name(ext)

    records: List[dict] = []
    rel = lambda p: p.relative_to(PROJECT_ROOT).as_posix()

    def extra_fields(s: Shot) -> dict:
        """记录里额外带的元信息，都是在合成噪声时会用到的。

        * `wb`     —— 该相机的白平衡增益（R/G/B 各被放大多少倍）
        * `mosaic` —— 原始拜耳平面（16-bit 灰度 PNG）的相对路径
        * `bayer`  —— CFA 相位名（"RG" / "GR" / "GB" / "BG"）

        后两个只在 `--save-mosaic` 时才有。详见 data/noise_model.py 顶部。
        """
        out: Dict[str, object] = {}
        if wb_map and s.camera in wb_map:
            out["wb"] = wb_rgb_gains(wb_map[s.camera])
        if mosaic_dir is not None and bayer_by_cam and s.camera in bayer_by_cam:
            p = mosaic_dir / f"{s.path.stem}.png"
            if p.exists():
                out["mosaic"] = rel(p)
                out["bayer"] = bayer_by_cam[s.camera]
        return out

    # 非 pair 类别：一条记录一张图
    for s in shots:
        if s.category in ("clean", "demo"):
            p = out_path(s)
            if p.exists():
                records.append({
                    "src": rel(p),
                    "cam": cam_map[s.camera],
                    "scene": s.scene_key,
                    "camera": s.camera,        # 报告/分组统计用
                    "category": s.category,
                    **extra_fields(s),
                })

    # pair：按 (相机, 场景) 分组配对
    pairs: Dict[str, List[Shot]] = {}
    for s in shots:
        if s.category == "pair":
            pairs.setdefault(s.scene_key, []).append(s)

    n_incomplete = 0
    for key, group in sorted(pairs.items()):
        group = sorted(group, key=lambda s: (s.iso, s.idx))
        gt = group[0]                       # ISO 最低的那张当 GT
        if len(group) < 2:
            n_incomplete += 1
            continue
        gt_p = out_path(gt)
        if not gt_p.exists():
            n_incomplete += 1
            continue
        for noisy in group[1:]:
            nz_p = out_path(noisy)
            if not nz_p.exists():
                n_incomplete += 1
                continue
            records.append({
                "src": rel(gt_p),
                "noisy_src": rel(nz_p),
                "cam": cam_map[gt.camera],
                "scene": gt.scene_key,
                "camera": gt.camera,
                "category": "pair",
                "iso": noisy.iso,          # 真实噪声图的 ISO，直接作为条件输入
                **extra_fields(gt),
            })

    if n_incomplete:
        print(f"  ⚠️  {n_incomplete} 个配对场景不完整（缺 GT 或缺某档 noisy），已跳过")

    return records


def write_previews(records: Sequence[dict], out_dir: Path, count: int) -> int:
    """导出人类可看的预览图（套 gamma），供 Photoshop 检查。

    ⚠️ 会**先清空**输出目录再写。之前是只往里追加 —— 换掉输入图之后，
    旧预览图会留在原地，而且文件名排得更靠前，于是被误当成"这一批的结果"。
    实测踩过：换了新 RAW 重新 build，读到的还是上一批的预览图，
    白白怀疑了半天模型。（同类坑还有 build_lmdb 覆盖写、train.jsonl 与
    train.lmdb.jsonl 混用 —— 都是"旧产物没清干净"。）
    """
    from data.imageio import read_linear, write_srgb_preview

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  清空旧的预览图目录：{out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for r in records[:count]:
        src = PROJECT_ROOT / r["src"]
        if not src.exists():
            continue
        write_srgb_preview(out_dir / f"{src.stem}_preview.tif", read_linear(src))
        n += 1
    return n


# --------------------------------------------------------------------------- #
# 子命令：calibrate
# --------------------------------------------------------------------------- #
def _accumulate_time_stats(paths: Sequence[str], crop: int):
    """逐像素的**时序**均值与方差（ddof=1）。流式实现，一次只持有一帧。

    为什么要流式：原来写成 `np.stack([...])`，而 read_raw_dn 一帧就是
    H×W×8 字节的 float64（5D3 22MP → 177MB/帧）。5D3 ISO1600 有 43 张、
    A7M3 ISO6400 有 50 张，一个 worker 的峰值轻松到 7GB+。
    4 个 worker 一起跑，内核 OOM killer 直接把 worker 干掉，
    主进程看到的就是 `BrokenProcessPool: A process ... terminated abruptly`。
    （别去查代码逻辑，那不是逻辑错误，是内存。）

    改成累加 Σx 和 Σx² 之后，每个 worker 只占固定的两三个平面，
    和帧数无关 —— 50 帧和 5 帧的内存一模一样。

    数值精度：Σx² 最大约 50×16383² ≈ 1.3e10，float64 的相对误差 1e-16，
    绝对误差 ~1e-6，而我们要的方差量级在 1e2~1e4 —— 完全够用。
    暗帧的均值虽小（~几百 DN）但帧数少，同样安全。

    返回 `(mean, var, n, eps)`。最后一个 `eps` 是**帧间总亮度的相对抖动**，
    不是噪声，是拍摄质量指标：见 `_flat_level_job` 与 cmd_calibrate 里的体检。
    """
    s1 = s2 = None
    n = 0
    fmeans: List[float] = []
    for p in paths:
        a = read_raw_dn(Path(p))
        if crop:
            a = a[crop:-crop, crop:-crop]
        if s1 is None:
            s1 = np.zeros(a.shape, dtype=np.float64)
            s2 = np.zeros(a.shape, dtype=np.float64)
        fmeans.append(float(a.mean()))       # 整帧平均亮度，用于体检光源
        # out= 原地累加，避免每帧再分配一个同样大的临时平面
        np.add(s1, a, out=s1)
        np.multiply(a, a, out=a)     # 复用 a 当临时缓冲，省一份平面
        np.add(s2, a, out=s2)
        n += 1

    if n < 2:
        raise ValueError("时序方差至少要 2 帧")

    mean = s1 / n
    # n·mean² 就是 s1·mean，展开后等价于 Σ(x-μ)²
    var = (s2 - s1 * mean) / (n - 1)

    # 纯噪声下帧均值只会抖动 ~1/sqrt(总像素数)，也就是万分之几；
    # 灯光在闪 / 曝光在变时会到百分之几。差别有三个数量级，
    # 所以这是判断平场能不能用的最灵敏的指标。
    fm = np.asarray(fmeans, dtype=np.float64)
    eps = float(fm.std(ddof=1) / fm.mean()) if fm.mean() > 0 else 0.0
    return mean, var, n, eps


def _dark_sigma_job(args: Tuple[List[str], int, int]) -> Tuple[float, int]:
    """一档 ISO 的读出噪声：逐像素时间标准差的均值（单位 DN）。"""
    paths, crop, _ = args
    _mean, var, n, _eps = _accumulate_time_stats(paths, crop)
    return float(np.sqrt(var).mean()), n


def _flat_level_job(args: Tuple[List[str], int]) -> Tuple[float, float, int, float]:
    """一个曝光档的平场统计量：平均亮度 + 平均时间方差，用于 PTC 拟合。"""
    paths, crop = args
    mu, var, n, eps = _accumulate_time_stats(paths, crop)

    # 只取中低亮度：接近饱和的点方差会失效，会严重带偏拟合。
    # 百分位在 1/8 抽样上算 —— 2200 万像素抽 280 万个，
    # 99 分位数早就收敛了，却能省掉一次 177MB 的全量排序拷贝。
    hi = np.percentile(mu.ravel()[::8], 99)
    ok = mu < 0.85 * hi
    if ok.sum() < 100:
        ok = np.ones_like(mu, dtype=bool)
    return float(mu[ok].mean()), float(var[ok].mean()), n, eps


def cmd_calibrate(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    shots, _ = scan_raws(raw_dir)
    if not shots:
        print(f"ERROR: {raw_dir} 里没有符合命名规范的 RAW")
        return 1

    darks = [s for s in shots if s.category == "dark"]
    flats = [s for s in shots if s.category == "flat"]
    print(f"标定输入：暗帧 {len(darks)} 张，平场 {len(flats)} 张")
    if not darks:
        print("ERROR: 没有暗帧，算不出读出噪声。见《数据拍摄清单》第二节。")
        return 1
    if not flats:
        print("ERROR: 没有平场，算不出系统增益。见《数据拍摄清单》第三节。")
        return 1

    cameras = sorted({s.camera for s in shots})
    params: Dict[str, Dict[str, List[float]]] = {}
    unstable_cams: List[str] = []

    with make_pool(args.workers) as pool:
        for cam in cameras:
            cam_darks = [s for s in darks if s.camera == cam]
            cam_flats = [s for s in flats if s.camera == cam]
            if not cam_darks or not cam_flats:
                print(f"\n⚠️  {cam}: 缺暗帧或缺平场，跳过")
                continue

            isos = sorted({s.iso for s in cam_flats} & {s.iso for s in cam_darks})
            print(f"\n{cam}: {len(isos)} 档 ISO")

            # ---- 读出噪声：每档 ISO 的暗帧逐像素时间标准差 ----
            dark_jobs = []
            for iso in isos:
                g = [s for s in cam_darks if s.iso == iso]
                if len(g) >= 3:
                    dark_jobs.append(([str(s.path) for s in g], args.crop, iso))
            sigma_map: Dict[int, float] = {}
            if dark_jobs:
                for (sigma, n), job in zip(
                    pool.map(_dark_sigma_job, dark_jobs), dark_jobs
                ):
                    sigma_map[job[2]] = sigma
                    print(f"  ISO{job[2]:<6} 暗帧 {n:>3} 张  σ_read = {sigma:.2f} DN")

            # ---- 系统增益：光子转移曲线 ----
            level_jobs = []
            for iso in isos:
                by_level: Dict[str, List[Shot]] = {}
                for s in cam_flats:
                    if s.iso == iso:
                        # 曝光档的分组键：文件名模式下是 ev03 这类档位名；
                        # 文件夹模式下没有档位名，就用**快门速度**——
                        # 不同的曝光档本来就是靠改快门实现的，天然一一对应。
                        by_level.setdefault(s.scene or s.shutter, []).append(s)
                levels = [(k, v) for k, v in sorted(by_level.items()) if len(v) >= 3]
                if len(levels) >= 2:
                    level_jobs.append((iso, [(k, [str(x.path) for x in v]) for k, v in levels]))

            per_cam: Dict[str, List[float]] = {}
            flicker_frac: List[float] = []
            for iso, levels in level_jobs:
                # 平场是计算量的大头（每档 ISO × 5 个曝光档），也丢进进程池
                flat_jobs = [([str(p) for p in v], args.crop) for _k, v in levels]
                # ⚠️ 必须 list() 一次：ProcessPoolExecutor.map 返回的是**一次性
                # 生成器**（不是 list），遍历两遍第二遍就是空的 —— 下面体检和
                # 拟合都要用 results，所以先固化下来。
                results = list(pool.map(_flat_level_job, flat_jobs))

                # ---- 平场体检：光照稳不稳 ----
                # 帧间总亮度抖动 eps 会让每个像素都跟着整体变，这部分方差
                # **和光子噪声长得一模一样**，会被 PTC 当成信号噪声吸收掉，
                # 表现就是 K 小得离谱、而且随机跳（不随 ISO 单调）。
                # 这里估「抖动贡献的方差 / 实测方差」，占比高就说明数据不可用。
                fracs = [(mu * eps) ** 2 / var
                         for mu, var, _n, eps in results if var > 0 and mu > 0]
                frac = float(np.median(fracs)) if fracs else 0.0
                flicker_frac.append(frac)

                pts = [(mu, var) for mu, var, _n, _e in results]
                if len(pts) < 2:
                    continue
                mus = np.array([m for m, _ in pts])
                vars_ = np.array([v for _, v in pts])
                slope, intercept = np.polyfit(mus, vars_, 1)
                if slope <= 0:
                    print(f"  ISO{iso:<6} ⚠️  拟合斜率为负（{slope:.3g}），这档跳过")
                    continue
                K = 1.0 / slope
                sig_fit = float(np.sqrt(max(intercept, 0.0)))
                sigma = sigma_map.get(iso, sig_fit)
                per_cam[str(iso)] = [round(K, 6), round(sigma, 4)]
                flag = "  ❌光照不稳" if frac > 0.3 else ""
                print(f"  ISO{iso:<6} K = {K:>9.5f} e-/DN   σ_read = {sigma:>6.2f} DN"
                      f"   （PTC 拟合截距 σ = {sig_fit:.2f}，"
                      f"抖动占方差 {frac*100:>5.1f}%）{flag}")

            if per_cam:
                params[cam] = per_cam

            if flicker_frac and float(np.median(flicker_frac)) > 0.3:
                unstable_cams.append(cam)
                print(f"\n  ❌ {cam} 的平场**光源不稳** —— 请重拍！")
                print(f"     PTC 拟合出的不是传感器噪声，是灯光在抖："
                      f"抖动方差中位数占 {np.median(flicker_frac)*100:.0f}%。")
                print(f"     症状：K 小到 0.0x 且不随 ISO 单调（校验器会报“标定写反了”）。")
                print(f"     修法见《数据拍摄清单》第三节：固定光圈/快门，用稳定连续光源，"
                      f"曝光到中灰以上。")

    # 光照不稳的相机直接剔除，不出现在产物里。
    # 理由：这些 K 是从灯光抖动里拟合出来的假数，
    #   1) 留着 noise_params.json 过不了校验，train.py 一加载就抛异常，等于把整个
    #      训练堵死；
    #   2) 就算绕过校验，合成的"噪声"也是错的，训出来的模型反而更差。
    # 宁可先只用可信的相机开训 —— 少一台就是少一组噪声特性，补上即可。
    if unstable_cams:
        print(f"\n⚠️  已把光照不稳的相机从结果中剔除：{', '.join(unstable_cams)}")
        print("    它们的 K 是从灯光抖动里拟合出来的假值（见上面的 ❌）。")
        print("    重拍平场后重跑本命令，它们会自动回到结果里。")
        for c in unstable_cams:
            params.pop(c, None)
        print(f"    本次可用相机：{', '.join(sorted(params)) or '（无）'}")

    if not params:
        print("\nERROR: 没有任何一台相机的参数可用。")
        print("  暗帧/平场的数量或命名有问题，或者所有相机的平场光照都不稳。")
        print("  先看上面每档 ISO 的「抖动占方差」，再去《数据拍摄清单》第三节对一遍。")
        return 1

    out_path = PROJECT_ROOT / args.out
    payload = {
        "_README": [
            "由 data/prepare_data.py calibrate 生成。",
            "每项为 [K, sigma_read]：K 是系统增益(e-/DN)，sigma_read 是读出噪声(DN)。",
            "噪声模型: sigma^2[DN] = DN/K + sigma_read^2",
            "校验：selfcheck.py 的「噪声参数校验器」项会检查 K 是否随 ISO 单调下降。",
        ],
        "cameras": [c for c in ("Canon5D3", "NikonZ6II", "SonyA7M3") if c in params]
                    or sorted(params),
        "params": params,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n已写出 {out_path}")
    try:
        np_obj = NoiseParams.load(out_path)
        print(np_obj.summary())
        print("✅ 参数校验通过（K 随 ISO 单调下降）")
    except ValueError as exc:
        print(f"\n⚠️  生成的参数没通过校验：\n{exc}")
        print("   这通常说明平场拍得有问题（太暗 / 过曝 / 光照不稳），回去看《数据拍摄清单》第三节。")
        return 1

    print("\n★ 最后一步（不能省）：目视验收")
    print("  把合成图和真实高 ISO 图并排 100% 放大对比，肉眼应难分伯仲。")
    print("  不像就说明标定或 ISP 有问题 —— 先修这个再训练。")
    return 0


# --------------------------------------------------------------------------- #
# 子命令：verify
# --------------------------------------------------------------------------- #
def estimate_shift(a: np.ndarray, b: np.ndarray, max_side: int = 512):
    """用相位相关估计 b 相对 a 的平移（原图像素）。返回 (dx, dy, 置信度)。"""
    import cv2

    def prep(x):
        # 用亮度当特征即可；转灰度前先按 [0,255] 缩放，避免相位相关数值退化
        g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)
        g -= g.mean()
        s = g.std()
        if s > 1e-8:
            g /= s
        scale = max(g.shape) / float(max_side)
        if scale > 1.0:
            g = cv2.resize(g, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        return g, scale

    ga, scale_a = prep(a)
    gb, _ = prep(b)
    h = min(ga.shape[0], gb.shape[0])
    w = min(ga.shape[1], gb.shape[1])
    ga, gb = ga[:h, :w], gb[:h, :w]

    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ga, gb, win)
    return dx * scale_a, dy * scale_a, float(response)


def _pair_check_job(job: Tuple[dict, float, float, int]) -> dict:
    rec, tol_bright, tol_shift, max_side = job
    from data.imageio import read_linear

    gt = read_linear(PROJECT_ROOT / rec["src"])
    nz = read_linear(PROJECT_ROOT / rec["noisy_src"])

    if gt.shape != nz.shape:
        return {"scene": rec["scene"], "iso": rec.get("iso"), "error":
                f"尺寸不一致 {gt.shape} vs {nz.shape}"}

    lum_gt = float((0.299 * gt[..., 0] + 0.587 * gt[..., 1] + 0.114 * gt[..., 2]).mean())
    lum_nz = float((0.299 * nz[..., 0] + 0.587 * nz[..., 1] + 0.114 * nz[..., 2]).mean())
    ratio = lum_nz / max(lum_gt, 1e-6)

    dx, dy, resp = estimate_shift(gt, nz, max_side)
    shift = float(np.hypot(dx, dy))

    return {
        "scene": rec["scene"],
        "iso": rec.get("_iso"),
        "lum_ratio": ratio,
        "shift_px": shift,
        "resp": resp,
        "bright_ok": abs(ratio - 1.0) <= tol_bright,
        "align_ok": shift <= tol_shift,
    }


def cmd_verify(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    shots, _ = scan_raws(raw_dir)
    if not shots:
        print(f"ERROR: {raw_dir} 里没有符合命名规范的 RAW")
        return 1

    lin_dir = PROJECT_ROOT / args.out / "linear"
    ext = args.ext
    print(f"配对检查（亮度容差 ±{args.tol_bright:.0%}，位移容差 {args.tol_shift:g}px）\n")

    pairs: Dict[str, List[Shot]] = {}
    for s in shots:
        if s.category == "pair":
            pairs.setdefault(s.scene_key, []).append(s)

    if not pairs:
        print("没有 pair 类别的照片，跳过配对检查。")
    jobs = []
    for key, group in sorted(pairs.items()):
        group = sorted(group, key=lambda s: (s.iso, s.idx))
        gt = group[0]
        gt_p = lin_dir / gt.out_name(ext)
        if not gt_p.exists():
            continue
        for noisy in group[1:]:
            nz_p = lin_dir / noisy.out_name(ext)
            if not nz_p.exists():
                continue
            jobs.append((
                {
                    "src": gt_p.relative_to(PROJECT_ROOT).as_posix(),
                    "noisy_src": nz_p.relative_to(PROJECT_ROOT).as_posix(),
                    "scene": key,
                    "_iso": noisy.iso,
                },
                args.tol_bright, args.tol_shift, args.max_side,
            ))

    bad_bright, bad_align, analysed = [], [], 0
    if jobs:
        with make_pool(args.workers) as pool:
            for r in pool.map(_pair_check_job, jobs):
                analysed += 1
                if "error" in r:
                    print(f"  ✗ {r['scene']:<18} {r['error']}")
                    continue
                flags = []
                if not r["bright_ok"]:
                    flags.append(f"亮度 {r['lum_ratio']:.3f}")
                    bad_bright.append(r)
                if not r["align_ok"]:
                    flags.append(f"位移 {r['shift_px']:.2f}px")
                    bad_align.append(r)
                mark = "✓" if not flags else "✗"
                extra = ("  ← " + " / ".join(flags)) if flags else ""
                print(f"  {mark} {r['scene']:<18} ISO{r['iso']:<6} "
                      f"亮度比 {r['lum_ratio']:.3f}  位移 {r['shift_px']:.2f}px"
                      f"  置信度 {r['resp']:.3f}{extra}")

    print(f"\n逐条检查了 {analysed} 对")

    # ---- 解码确定性（铁律 #5）----
    det_ok = True
    if args.determinism_check:
        from data.imageio import read_linear
        sample = next((s for s in shots if s.category in ("clean", "pair")), None)
        if sample:
            # 同理：这里只用到样片自己那台相机的白平衡
            wb_map = resolve_white_balance([sample], args.wb_mode, args.wb)
            a = decode_linear(sample.path, wb_map[sample.camera])
            b = decode_linear(sample.path, wb_map[sample.camera])
            det_ok = bool(np.array_equal(a, b))
            print(f"\n解码确定性（同一文件解码两次逐像素一致）：{'✅ 通过' if det_ok else '❌ 不一致'}")

    print()
    problems = 0
    if bad_bright:
        problems += 1
        print(f"❌ {len(bad_bright)} 对亮度不一致 —— 曝光量没对齐")
        print("   规则：ISO 翻 2 档，快门就要快 2 档。GT 和 noisy 的总曝光量必须相同。")
        for r in bad_bright[:3]:
            print(f"     {r['scene']} ISO{r['iso']}: 比值 {r['lum_ratio']:.3f}")
    if bad_align:
        problems += 1
        print(f"❌ {len(bad_align)} 对有位移 —— 这组配对数据基本报废")
        print("   对策：三脚架、关防抖、遥控快门、电子前帘、反光板预升（5D3）。")
        for r in bad_align[:3]:
            print(f"     {r['scene']} ISO{r['iso']}: {r['shift_px']:.2f}px")
    if not det_ok:
        problems += 1
        print("❌ 解码不确定 —— 检查 postprocess 参数里有没有随机项")

    if problems:
        print("\n⚠️  有问题先修，不要带着坏数据往下走。")
        return 1
    print("✅ 全部通过。")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RAW → 线性 sRGB 数据管线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--raw-dir", required=True, help="RAW 文件目录")
        p.add_argument("--workers", type=int, default=4,
                       help="并行进程数。**0 = 单进程**（调试用；"
                            "多进程出问题时用它兜底）。"
                            "每个 worker 约占 3~4 个全画幅平面（5D3 约 0.7GB），"
                            "内存不够就调小")
        p.add_argument("--wb-mode", choices=["fixed-per-camera", "as-shot", "explicit"],
                       default="fixed-per-camera", help="白平衡策略（默认最安全）")
        p.add_argument("--wb", default=None,
                       help="explicit 模式下每台相机的 WB，JSON，如 "
                            "'{\"NikonZ6II\":[2.1,1.0,1.45,1.0]}'")

    p = sub.add_parser("build", help="RAW → 线性 TIFF/PNG + 索引")
    add_common(p)
    p.add_argument("--out", default="data/real", help="输出目录（相对项目根）")
    p.add_argument("--ext", default=".tif", choices=[".tif", ".png"],
                   help="输出格式。tif 在 PS 里更顺手")
    p.add_argument("--categories", nargs="+", default=["clean", "pair", "demo"],
                   help="要解码的类别（dark/flat 留给 calibrate，不必解码）")
    p.add_argument("--pair-gap", type=float, default=15.0,
                   help="pair 平铺时按拍摄时间自动分组：相邻两张间隔超过这么多秒"
                        "就算换场景。组切碎了调大，不同场景并到一起了调小")
    p.add_argument("--val-frac", type=float, default=0.1, help="验证集场景占比")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="已存在的输出文件跳过（支持中断后续跑）")
    p.add_argument("--make-preview", action="store_true",
                   help="额外导出少量套过 gamma 的预览图，供 Photoshop 检查")
    p.add_argument("--preview-count", type=int, default=8)
    p.add_argument("--save-mosaic", action="store_true",
                   help="额外导出原始拜耳平面（16-bit 灰度 PNG，约 RGB 的 1/3 体积）。"
                        "开了才能用 data.synthesis=mosaic（拜耳域合成噪声）——"
                        "空间相关和通道结构物理正确，不用手工拟合参数")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("calibrate", help="暗帧 + 平场 → noise_params.json")
    add_common(p)
    p.add_argument("--out", default="noise_params.json")
    p.add_argument("--crop", type=int, default=32,
                   help="边缘裁掉的像素（避开暗角与坏点），0 表示不裁")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("verify", help="配对数据的亮度与对齐自检")
    add_common(p)
    p.add_argument("--out", default="data/real", help="build 的输出目录")
    p.add_argument("--ext", default=".tif")
    p.add_argument("--tol-bright", type=float, default=0.10, help="亮度比容差")
    p.add_argument("--tol-shift", type=float, default=1.0, help="位移容差（像素）")
    p.add_argument("--max-side", type=int, default=512, help="相位相关用的降采样边长")
    p.add_argument("--determinism-check", action="store_true", default=True)
    p.set_defaults(func=cmd_verify)

    return ap


def main() -> int:
    args = build_parser().parse_args()

    # ★ 必须在任何进程池创建之前调用，Linux 上尤其重要（rawpy + fork = 死锁）
    method = init_multiprocessing()
    if args.workers > 0 and method != "spawn":
        print(f"⚠️  多进程启动方式 = {method}（不是 spawn）。")
        print("    Linux 上 rawpy + fork 会死锁（表现为 BrokenProcessPool）。")
        print("    建议加 `--workers 0` 单进程跑。")

    if args.wb is not None and isinstance(args.wb, str):
        try:
            args.wb = json.loads(args.wb)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --wb 不是合法 JSON: {exc}")
            return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n中断。已完成的解码结果保留，重跑会自动跳过（--skip-existing）。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
