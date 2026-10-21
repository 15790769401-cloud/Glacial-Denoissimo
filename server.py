"""Denoissimo Demo 后端 —— 把 infer.py 的 Python 接口包成给 HTML 调的 HTTP 接口。

HTML/JS 不能 import Python，所以中间必须有这一层。这一层**不含任何算法**：
收 HTTP → 按后缀选通道（JPEG / RAW）→ 调 infer.py 里已经冻结的函数 → 把结果编成 JSON。
RAW 的双模式设计见《前端·JPG与RAW双模式》。

两个接口：

    GET  /api/info      相机列表 + 各档 ISO + **完整噪声参数表**  （填下拉框 / 本地预览参数）
    POST /api/denoise   上传图 + 相机 + ISO → 降噪后的图          （主接口）

> 旧的 `GET /api/noise-params` 已合并进这两个接口：
>   · 想**上传前**预览 K/σ → 用 `/api/info` 返回的 `noise_table` 在浏览器本地算，不再发请求
>   · 想**知道实际用了哪组** K/σ → `/api/denoise` 的响应里已经带了
> 这样"预览值"和"实际值"永远来自同一份表，不会再出现两边对不上。

跑（在项目根目录，和 demo_model.pth 同级）：

    pip install fastapi uvicorn python-multipart
    pip install rawpy exifread          # 只有 RAW 模式需要
    python server.py
    # 或： uvicorn server:app --host 127.0.0.1 --port 8000

然后浏览器开 http://127.0.0.1:8000/

换 checkpoint：  DENOISSIMO_CKPT=runs/xxx/last.pth python server.py
"""

from __future__ import annotations

import base64
import json
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# ⚠️ 必须在 import torch 之前设置，和 infer.py 保持一致
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from data.noise_model import NoiseParams  # noqa: E402
from infer import (  # noqa: E402
    denoise_tiled,
    read_exif_camera_iso,
    read_image,
    write_image,
)

# RAW 双模式依赖 infer.py 里的三个符号。**必须可选导入** ——
# 旧版 infer.py 没有它们，写死在顶部会让整个服务（连 JPEG）都起不来。
_RAW_EXTS_FALLBACK = {".arw", ".cr2", ".cr3", ".nef", ".nrw",
                      ".dng", ".raf", ".rw2", ".orf"}
try:
    from infer import (  # noqa: E402
        RAW_EXTS as _RAW_EXTS_FROM_INFER,
        read_raw_as_srgb,
        read_raw_camera_iso,
    )
    RAW_EXTS = set(_RAW_EXTS_FROM_INFER)
    RAW_SUPPORTED = True
except ImportError:
    RAW_EXTS = _RAW_EXTS_FALLBACK
    read_raw_as_srgb = None          # type: ignore[assignment]
    read_raw_camera_iso = None       # type: ignore[assignment]
    RAW_SUPPORTED = False
from models import build_model  # noqa: E402
from models.conditioner import resolve_noise_params  # noqa: E402
from utils.train_utils import load_checkpoint  # noqa: E402

CKPT = Path(os.environ.get("DENOISSIMO_CKPT", ROOT / "demo_model.pth"))
NOISE_JSON = Path(os.environ.get("DENOISSIMO_NOISE", ROOT / "noise_params.json"))

# RAW 模式的白平衡。
#   不设 → 用文件内嵌的 as-shot WB（前提：演示 RAW 和训练数据是同一个 WB 设置）
#   设了 → 强制用这组，格式 "R,G,B,G2"（rawpy 的 user_wb）
#          值从 prepare_data build 日志里的 fixed-per-camera 抄，例：
#          DENOISSIMO_WB="2.784,1,1.512,1"
_WB = os.environ.get("DENOISSIMO_WB", "").strip()
FORCE_WB = tuple(float(x) for x in _WB.split(",")) if _WB else None


# --------------------------------------------------------------------------- #
# 启动时做一次：加载模型 + 读标定表
# --------------------------------------------------------------------------- #
def load_engine():
    if not CKPT.exists():
        raise SystemExit(
            f"找不到 checkpoint: {CKPT}\n"
            f"把 demo_model.pth 放到 {ROOT}，或用 DENOISSIMO_CKPT=... 指定路径。"
        )
    # weights_only=False：checkpoint 里除了张量还有 config dict。
    # torch>=2.6 默认 weights_only=True，遇到非基础类型会直接拒载 —— 必须显式关掉。
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])          # ★ 配置在 checkpoint 里
    load_checkpoint(CKPT, model, use_ema=True)   # ★ EMA 不能省
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    return model, device, amp


print(f"[boot] checkpoint = {CKPT}")
MODEL, DEVICE, AMP = load_engine()
NOISE = NoiseParams.load(NOISE_JSON)
print(f"[boot] device = {DEVICE}  amp = {AMP}")
if DEVICE.type != "cuda":
    print("[boot] ⚠️ 未检测到 CUDA，模型将用 CPU 推理（会很慢）。")
    print("[boot]    原因：当前 torch 是 CPU 编译版（Windows 下 pip install torch 默认如此）。")
    print("[boot]    启用 GPU（RTX 5080，Blackwell sm_120）请重装 CUDA 版 torch：")
    print("[boot]      .venv\\Scripts\\python.exe -m pip install torch "
          "--index-url https://download.pytorch.org/whl/cu132")
    print("[boot]    装完重启服务，这里应显示 device = cuda。")
print("[boot] RAW 双模式 = " +
      ("可用" if RAW_SUPPORTED else
       "⚠️ 不可用 —— 同目录的 infer.py 是旧版，只能用 JPEG"))
if RAW_SUPPORTED:
    print("[boot] RAW 白平衡 = " +
          (f"强制 {FORCE_WB}" if FORCE_WB else "文件内嵌 as-shot（需与训练一致）"))
print(NOISE.summary())

app = FastAPI(title="Denoissimo Demo", version="1.0")

# 同源访问（推荐：由本服务托管 index.html）时用不到 CORS。
# 但如果前端是双击 file:// 打开的，Origin 是 "null"，没有这段就会被浏览器拦掉。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _encode_png(rgb01: np.ndarray) -> bytes:
    """(H,W,3) float32 [0,1] → PNG 字节。走项目自己的 write_image，保证和别处一致。"""
    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp = Path(name)
    try:
        write_image(tmp, rgb01, bit_depth=8)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def _png_data_url(rgb01: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(
        _encode_png(rgb01)).decode("ascii")


def _save_upload(raw: bytes, filename: Optional[str]) -> Path:
    """把上传的字节落盘 —— read_image() 是按路径读的。"""
    suffix = Path(filename or "upload.png").suffix or ".png"
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    tmp = Path(name)
    tmp.write_bytes(raw)
    return tmp


# --------------------------------------------------------------------------- #
# 接口 1：相机 / ISO 列表 + 完整噪声参数表   →  models.NoiseParams
# --------------------------------------------------------------------------- #
def _noise_table() -> dict:
    """整张标定表打成 JSON：{相机: {ISO: [K, sigma_read]}}。

    只有 30 个数字，几百字节 —— 随 /api/info 一次发下去，
    前端就能在本地算出任何 (相机, ISO) 对应的参数，不用再发请求。
    """
    out: dict = {}
    for cam in NOISE.cameras:
        cid = NOISE.cam_id(cam)
        per_iso = {}
        for iso in NOISE.iso_list(cid):
            k, s = NOISE.get(cid, iso)
            per_iso[int(iso)] = [round(float(k), 8), round(float(s), 4)]
        out[cam] = per_iso
    return out


@app.get("/api/info")
def api_info():
    return {
        "ok": True,
        "cameras": [
            {"name": cam, "isos": NOISE.iso_list(NOISE.cam_id(cam))}
            for cam in NOISE.cameras
        ],
        # ★ 前端本地预览 K/σ 用这个，不要再调接口
        "noise_table": _noise_table(),
        "device": str(DEVICE),
        "amp": str(AMP),
        "ckpt": CKPT.name,
    }


# --------------------------------------------------------------------------- #
# 接口 2：降噪（主接口）  →  infer.denoise_tiled
#          ★ 参数只在这里解析一次，响应把它原样带回去 —— 前端以后端返回的为准
#          ★ 用 SSE 流式返回：先推「阶段 + 真实分块进度」，最后推 done 结果。
#            前端进度条/加载文字跟着真实进度走，不再是写死的假动画。
# --------------------------------------------------------------------------- #
def _sse(payload: dict) -> str:
    """把一条事件编成 Server-Sent Events 的 data 行。"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.post("/api/denoise")
async def api_denoise(
    file: UploadFile = File(...),
    camera: str = Form(""),        # "" = 自动读 EXIF
    iso: int = Form(3200),
    tile: int = Form(512),
    overlap: int = Form(32),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传的是空文件")

    # overlap 必须 < tile，否则分块窗口退化；契约里承诺的是 400，别让它跑进模型再炸
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise HTTPException(
            status_code=400,
            detail=f"tile={tile}、overlap={overlap} 不合法：要求 0 ≤ overlap < tile")

    # ① 走哪条线：按**文件名后缀**判断（前端必须保留原始文件名，别重命名成 upload.png）
    suffix = Path(file.filename or "").suffix.lower()
    mode = "raw" if suffix in RAW_EXTS else "jpg"
    if mode == "raw" and not RAW_SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail="服务端的 infer.py 是旧版，没有 RAW 解码函数"
                   "（read_raw_as_srgb）。请更新 infer.py，或改用 JPEG 输入。")

    # 处理逻辑放进后台线程，事件经队列喂给 SSE —— 这样分块回调能实时往外推进度
    q: "queue.Queue[dict]" = queue.Queue()

    def emit(**kw: object) -> None:
        q.put(kw)

    def worker() -> None:
        tmp: Optional[Path] = None
        try:
            t0 = time.perf_counter()      # 解码也算进去 —— 前端显示端到端耗时
            emit(type="stage", stage="upload", progress=2, label="正在上传照片")

            tmp = _save_upload(raw, file.filename)
            emit(type="stage", stage="exif", progress=6, label="正在读取相机与 ISO 信息")

            # ② EXIF：RAW 用 exifread（Pillow 读不了 RAW 内嵌 EXIF）。
            #    用下标取值而不是解包 —— 万一 infer.py 那边多返回字段会炸，
            #    这里炸了不该让整单失败。
            try:
                got = (read_raw_camera_iso if mode == "raw" else
                       read_exif_camera_iso)(tmp)
                exif_cam, exif_iso = got[0], got[1]
            except Exception:             # noqa: BLE001  读不到 EXIF 不该让整单失败
                exif_cam, exif_iso = None, None

            use_cam = camera.strip() or exif_cam
            use_iso = int(iso) if iso else (exif_iso or 3200)

            K, sigma, source, used_cam = resolve_noise_params(use_cam or None, use_iso, NOISE)

            # ③ 解码 —— 两条线在这里分岔
            emit(type="stage", stage="decode", progress=10, label="正在解码图像")
            if mode == "raw":
                # 和训练**完全同一条管线**：prepare_data 那 9 个 rawpy 参数 + gamma 2.2。
                # 白平衡用文件里的 as-shot WB（前提：演示 RAW 和训练数据同一 WB 设置）。
                try:
                    rgb = read_raw_as_srgb(tmp, wb=FORCE_WB)
                except ImportError as exc:
                    emit(type="error", detail=f"服务端没装 rawpy，读不了 RAW：{exc}")
                    return
                except Exception as exc:  # noqa: BLE001
                    emit(type="error", detail=f"RAW 解码失败：{exc}")
                    return
            else:
                try:
                    rgb = read_image(tmp)
                except (FileNotFoundError, ValueError) as exc:
                    emit(type="error", detail=f"读不了这张图：{exc}")
                    return

            if rgb.ndim != 3 or rgb.shape[2] != 3:
                emit(type="error", detail=f"解出来是 {rgb.shape}，期望 (H,W,3)")
                return

            # ④ RAW 的"原图"要在**推理前**存下来 —— 就是喂给模型的那张
            input_preview = _png_data_url(rgb) if mode == "raw" else None

            chw = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]

            # ⑤ 降噪（OOM 自动降 tile 重试一次）。进度：10%→94% 映射到分块完成比例。
            def on_tile(done: int, total: int) -> None:
                pct = 10 + int(round(84.0 * done / total)) if total else 94
                emit(type="progress", stage="denoise", progress=pct,
                     done=done, total=total, label=f"正在降噪 {done}/{total} 块")

            used_tile = tile
            try:
                out = denoise_tiled(MODEL, chw, K, sigma, tile=tile, overlap=overlap,
                                    device=DEVICE, amp_dtype=AMP, progress_cb=on_tile)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                used_tile = 256
                emit(type="stage", stage="denoise", progress=12,
                     label="显存不足，改用 256 分块重试")
                out = denoise_tiled(MODEL, chw, K, sigma, tile=used_tile, overlap=16,
                                    device=DEVICE, amp_dtype=AMP, progress_cb=on_tile)

            emit(type="stage", stage="encode", progress=95, label="正在生成结果图")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # ⑥ denoise_tiled 返回的就是 CPU 上的 (1,3,H,W)
            den = out[0].permute(1, 2, 0).numpy()
            b64 = base64.b64encode(_encode_png(den)).decode("ascii")

            resp = {
                "ok": True,
                "mode": mode,                              # "raw" | "jpg"
                "image": f"data:image/png;base64,{b64}",   # 直接塞 <img src>
                "camera": use_cam,
                "used_camera": used_cam,
                "iso": use_iso,
                "K": K,
                "sigma_read": sigma,
                "source": source,          # calibrated 精确 | brand 同品牌近似 | median 中位数
                "exif_camera": exif_cam,                   # 让 UI 能显示"识别到了什么"
                "exif_iso": exif_iso,
                "width": int(den.shape[1]),
                "height": int(den.shape[0]),
                "tile": used_tile,
                "elapsed_ms": round(elapsed_ms, 1),
            }
            if input_preview:
                resp["input_preview"] = input_preview      # ★ raw 模式下左图用它
            emit(type="done", progress=100, label="降噪完成", **resp)
        except Exception as exc:          # noqa: BLE001
            emit(type="error", detail=f"处理失败：{exc}")
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            msg = q.get()
            yield _sse(msg)
            if msg.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# 托管 index.html —— 同源，彻底没有 CORS 问题。必须放在最后。
# --------------------------------------------------------------------------- #
WEB = ROOT / "web"
if WEB.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
