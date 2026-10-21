"""配置加载 + 项目路径 bootstrap。

**为什么要有这个文件**：项目里既有 `python train.py`（根目录直接跑）也有
`python data/make_dummy_data.py`（子目录跑）。后者的 sys.path[0] 是 `data/`
而不是项目根，`import models` 之类会直接失败。所有入口脚本开头调用
``ensure_project_on_path()`` 就能消除这一整类"跑不起来"的困惑。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent

__all__ = [
    "PROJECT_ROOT",
    "ensure_project_on_path",
    "load_config",
    "save_config",
    "deep_update",
    "cfg_get",
]


def ensure_project_on_path() -> Path:
    """把项目根目录放到 sys.path 首位。幂等。"""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: str | Path) -> dict:
    """读 YAML 配置。相对路径按项目根解析。

    **遇到重复的键会直接报错**（见 `_strict_loader`）。
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("需要 pyyaml：uv pip install pyyaml") from exc

    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到配置文件 {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=_strict_loader())
    except yaml.YAMLError as exc:
        raise ValueError(
            f"配置文件 {p} 解析失败：\n{exc}\n"
            "（最常见的两种原因：缩进不对、同一个键写了两次）"
        ) from exc

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件 {p} 的顶层必须是一个 mapping，收到 {type(cfg).__name__}")
    return cfg


_STRICT_LOADER = None


def _strict_loader():
    """构造一个「遇到重复键就报错」的 YAML loader（懒加载，只建一次）。

    PyYAML 默认的行为是「后面的覆盖前面的」，而且**不报任何错**。
    这会让"我明明改了配置却没生效"变成最难查的一类问题 ——
    实际踩过：往 `model:` 段加了一行 `pretrained: weights/xxx.pth`，
    但没删掉模板里原有的 `pretrained: null`。因为 null 在下面，
    路径被它静默覆盖，日志只说"未配置 model.pretrained"，配置文件看起来完全正常。
    """
    global _STRICT_LOADER
    if _STRICT_LOADER is not None:
        return _STRICT_LOADER

    import yaml

    class _UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            seen = {}
            for key_node, _value in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    dup = key in seen
                except TypeError:  # 键不可哈希（YAML 里极少见）
                    dup = False
                if dup:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"重复的键 {key!r}：第 {seen[key]} 行已经定义过一次。\n"
                        f"    YAML 会用后一个覆盖前一个，所以第一次那个不会生效。\n"
                        f"    删掉其中一行即可。",
                        key_node.start_mark,
                    )
                seen[key] = key_node.start_mark.line + 1
            return super().construct_mapping(node, deep=deep)

    _STRICT_LOADER = _UniqueKeyLoader
    return _STRICT_LOADER


def save_config(cfg: Mapping, path: str | Path) -> Path:
    """把配置随 checkpoint 一起存下来。

    一周后没人记得这组超参是什么 —— 不存配置的 checkpoint 是没法复现的。
    """
    import yaml

    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)
    return p


def deep_update(base: Mapping, override: Mapping) -> dict:
    """递归合并，override 覆盖 base。返回新 dict，不改原对象。"""
    out = copy.deepcopy(dict(base))
    for k, v in dict(override).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cfg_get(cfg: Mapping, key: str, default: Any = None, *, required: bool = False) -> Any:
    """取配置项，缺失时给出清晰的报错而不是 KeyError。"""
    if key in cfg:
        return cfg[key]
    if required:
        raise KeyError(
            f"配置缺少必需项 {key!r}。当前可用的键: {sorted(cfg.keys())}"
        )
    return default
