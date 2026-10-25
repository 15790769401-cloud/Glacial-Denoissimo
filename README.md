# Denoissimo Demo

**一个模型，处理三台相机、十档 ISO 的图像降噪。**

条件输入不是相机型号，而是该相机在该 ISO 下的物理噪声参数 `(K, σ_read)`。
单一模型覆盖佳能 5D3 / 尼康 Z6II / 索尼 A7M3 共 30 组噪声特性，无需为每种组合单独训练。

本目录是可直接运行的最小演示系统：FastAPI 后端 + 单页 Web 界面。

---

## 快速开始

需要 **Python 3.12** 和一张 NVIDIA 显卡（无显卡也能跑，但在 CPU 上会很慢）。

```bash
# 1. 建环境
python -m venv .venv

# 2. 装 torch —— 必须用 CUDA 12.8 的源
#    （RTX 50 系是 Blackwell sm_120，其他源的轮子跑不起来）
.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 装其余依赖
.venv/bin/python -m pip install -r requirements.txt

# 4. 确认 GPU 可用（应输出 (12, 0)）
.venv/bin/python -c "import torch; print(torch.cuda.get_device_capability())"

# 5. 启动
.venv/bin/python server.py
```

Windows PowerShell 把 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。

浏览器打开 **http://127.0.0.1:8000/**

启动日志：

```
[boot] checkpoint = .../demo_model.pth
[boot] device = cuda  amp = torch.bfloat16
[boot] RAW 双模式 = 可用
```

---


## 目录结构

```
.
├── server.py            HTTP 接口（不含算法，只做收请求 → 调推理 → 编 JSON）
├── infer.py             推理入口：分块推理、EXIF、RAW/JPEG 双路径
├── config.py            配置
├── models/              Restormer 骨干 + 噪声参数条件化模块
├── data/                噪声模型、编解码（只含代码，不含任何图像）
├── utils/               检查点加载、指标
├── web/index.html       前端（单文件，无依赖）
├── noise_params.json    三台相机 × 10 档 ISO 的标定结果
└── requirements.txt
```

---

## 两个接口

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/info` | 相机列表 + `noise_table`（30 组 K/σ，供前端本地预览） |
| `POST` | `/api/denoise` | 上传图 + 相机 + ISO → 降噪结果 |

`POST /api/denoise` 的响应包含：`image`（data URL）、`mode`（`raw` / `jpg`）、
`K` / `sigma_read`（实际使用的参数）、`source`（`calibrated` / `brand` / `median`）、
RAW 模式下额外返回 `input_preview`（浏览器无法直接显示 RAW）。

接口的完整契约见前端代码中的调用处。

---

## 输入格式

| | 输入 | 说明 |
|---|---|---|
| **专业模式** | `.ARW` `.CR2` `.CR3` `.NEF` `.DNG` 等 | 与训练完全一致的解码管线，效果最好 |
| **快速模式** | `.JPG` `.PNG` `.TIF` 等 | 经过机内降噪、锐化与 JPEG 量化，效果有折扣 |

RAW 按**文件名后缀**识别，上传时请保留原始文件名。

---

## 已知限制

- **只标定过三台相机**：佳能 5D3、尼康 Z6II、索尼 A7M3。
  其他机型会走品牌级或中位数兜底，结果中的 `source` 字段会标为 `brand` / `median`，
  **属于近似值**。
- **不声称支持任意相机**，也未在未标定的机型上验证过。
- 模型训练使用的是按标定参数合成的噪声。在相机直出 JPEG 上，
  平坦区噪声下降幅度约为 RAW 输入的**一半** —— 这正是本演示把 RAW 作为主线、
  JPEG 作为兼容模式的原因。
- 24MP RAW 全分辨率推理约 20–30 秒（RTX 5060 Ti 8GB）。

---

## 可选环境变量

| 变量 | 作用 |
|---|---|
| `DENOISSIMO_CKPT` | 指定模型文件路径 |
| `DENOISSIMO_NOISE` | 指定标定表路径 |
| `DENOISSIMO_WB` | 强制 RAW 白平衡，格式 `"R,G,B,G2"`（默认用文件内嵌的 as-shot） |

---

## 致谢

骨干网络为 [Restormer](https://github.com/swz30/Restormer)（CVPR 2022, Zamir et al.），
加载其官方 SIDD 预训练权重。

---

## 效果展示 
高ISO（2000）：
<img width="1914" height="871" alt="Image_1792338336283_798" src="https://github.com/user-attachments/assets/b89c72b7-125b-449f-8f6a-aac8436614e8" />
<img width="1912" height="872" alt="Image_1792338337727_902" src="https://github.com/user-attachments/assets/86854ff1-4886-4635-8b4f-c57ee5f73584" />

极高ISO（51200）
<img width="2736" height="1689" alt="Image_1792135501980_94" src="https://github.com/user-attachments/assets/5f9415c2-8fed-43bd-89a1-64493e068bbb" />
<img width="2736" height="1689" alt="Image_1792135503939_291" src="https://github.com/user-attachments/assets/19a31d2d-c359-4fd4-a43c-65b84c5dcd11" />


