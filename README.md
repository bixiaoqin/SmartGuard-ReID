# SmartGuard-ReID

面向智慧安防的行人重识别（ReID）系统 —— 天津市大学生创新训练计划项目《面向智慧安防的行人重识别关键算法研究》配套原型系统。

支持三种核心场景的行人检索演示：**常规无约束 / 遮挡 / 换衣**。

## 架构

```
视频流/图片 → YOLOv8 行人检测 → ReID 引擎提取特征 → 余弦相似度检索 → Top-K 排名展示
```

- **检测**：`detector.py`（YOLOv8n，ultralytics）
- **视频流**：`stream.py`（视频文件 / 摄像头 / RTSP 统一封装，逐帧 YOLO + ByteTrack 编号）
- **特征引擎**：`extractor.py`（当前为 FastReID R50 旧引擎；算法组的新引擎只需替换此文件内部的模型，接口 `extract(img_bgr) -> 特征向量` 不变）
- **Web 原型**：`app.py`（Flask）+ `templates/` + `static/`（上传图片 / 监控视频框选 → 跨镜头检索）
- **主流程**：`retrieve.py`（一键检索 + rank list 可视化，检索逻辑也在此）
- **全局配置**：`config.py`（所有路径、设备参数集中管理）

## 快速开始（组员从零搭建）

### 1. 克隆仓库

```bash
git clone https://github.com/bixiaoqin/SmartGuard-ReID.git
cd SmartGuard-ReID
```

### 2. 下载仓库外的大文件

权重、FastReID 源码、数据图片因体积/版权原因不随 git 仓库分发，运行：

```bash
python scripts/download_assets.py
```

脚本会自动完成：

| 资源 | 来源 | 落位 |
|---|---|---|
| `yolov8n.pt`（检测权重） | ultralytics 官方发布 | `weights/` |
| `market_bot_R50.pth`（ReID 旧引擎权重，约 294MB） | FastReID 官方 MODEL_ZOO | `weights/` |
| `fast-reid/`（FastReID 源码） | github.com/JDAI-CV/fast-reid | 项目根目录 |

> 若某项下载失败，也可手动获取后放到对应位置（见上表）。

### 3. 配置 Python 环境

需要 Anaconda，环境名 `reid`：

```bash
conda create -n reid python=3.10 -y
conda activate reid
pip install -r requirements.txt
```

Windows 下也可直接双击 `setup.bat`（在 Anaconda Prompt 中运行）。

### 4. 验证

```bash
python retrieve.py
```

输出 Top-5 检索结果并生成 `outputs/` 下的 rank list 图，即环境搭建成功。

## 数据集下载（算法组训练 / 三场景测试用）

| 场景 | 数据集 | 下载地址 | 放置位置 |
|---|---|---|---|
| 常规 | Market-1501 | Kaggle：`kaggle.com/datasets/pengcw1/market-1501` | `data/datasets/Market-1501/` |
| 遮挡 | Occluded-Duke | `github.com/lightas/Occluded-DukeMTMC-Dataset` | `data/datasets/Occluded-Duke/` |
| 换衣 | PRCC | 官方申请页：`www.isee-ai.cn/~yangqize/clothing.html` | `data/datasets/PRCC/` |

**下载后怎么做：**

1. **Market-1501**：解压后应得到 `bounding_box_train/`、`bounding_box_test/`、`query/` 等文件夹，整个数据集文件夹放进 `data/datasets/Market-1501/`
2. **Occluded-Duke**：此仓库**只提供文件名列表，不含图片**——先下载原始 DukeMTMC-reID（Kaggle：`kaggle.com/datasets/igorkrashenyi/dukemtmc-reid`），再用其仓库里的 `convert_duke_to_occduke.py` 脚本转换，生成的 `bounding_box_train/`、`bounding_box_test/`、`query/` 放进 `data/datasets/Occluded-Duke/`
3. **PRCC**：在官方页面提交申请（通常通过百度网盘/邮件发放），解压后的 `prcc/`（含 `train/`、`test/`、`query/`）放进 `data/datasets/PRCC/`
4. 数据集只放本地，**不要提交到 git**（`.gitignore` 已排除 `data/datasets/`，防止误传版权数据）

系统默认演示数据（`data/query.jpg` + `data/gallery/`，19 张，约 1MB）**随仓库分发**：行人 0001 从摄像头 c1 发起查询，图库是他在 c2~c6 其他机位的画面 + 10 个干扰行人 —— 克隆即可演示"跨镜头检索"。

## 运行 Web 原型系统（中期演示）

```bash
conda activate reid
python app.py        # 浏览器打开 http://127.0.0.1:5000
```

两条入口：

- **入口 A（监控视频跨镜头检索）**：选择 CAMPUS 多机位监控视频（本机 `data/config.py` 里 `CAMPUS_VIDEO_DIR` 指向的目录，不入库）→ 抓取画面并 YOLO 检测 → **在画面上点击目标行人** → 裁成 query → 在其他机位的图库中检索 → 展示 rank list
- **入口 B（上传图片检索）**：上传一张行人图片，自动检测并检索 Top-5

命令行自测视频流：

```bash
python stream.py --video "D:/dataset/database_path/multiview-database/CAMPUS/Garden2/view-HC1.mp4" --seek 30 --track
```

> CAMPUS 多机位监控视频与跨镜头图库（`data/campus/`）体积大、有版权，只放本地不入库；组员克隆仓库后仍可用入口 B + Market 子集完整体验检索流程。

## 目录结构

```
SmartGuard-ReID/
├── config.py            # 全局配置（路径、设备）
├── app.py               # Flask Web 原型（上传图片 / 监控视频框选 → 跨镜头检索）
├── stream.py            # 视频流封装（视频文件/摄像头/RTSP + YOLO + ByteTrack）
├── retrieve.py          # 一键检索：查询图 → gallery 检索 Top-K + rank list 可视化
├── retrieve.py / retrieve.bat   # 一键检索脚本（命令行 / Windows 双击）
├── detector.py          # YOLOv8 行人检测封装
├── extractor.py         # ReID 特征引擎（唯一换引擎入口）
├── templates/ static/   # Web 原型页面
├── weights/             # 模型权重（脚本下载，不入库）
├── fast-reid/           # FastReID 源码（脚本克隆，不入库）
├── data/
│   ├── query.jpg        # 默认查询图（Market 行人0001 @ 摄像头c1，随仓库分发）
│   ├── gallery/         # 默认图库（行人0001其他机位 + 干扰项，随仓库分发）
│   ├── campus/          # CAMPUS 跨镜头图库（本地，不入库）
│   ├── datasets/        # 完整数据集（本地，不入库）
│   └── scenarios/       # 三场景测试集（生成产物）
├── outputs/             # 生成的 rank list 等结果
├── scripts/             # 资源下载、三场景测试集生成工具
├── tests/               # 单项测试脚本
└── docs/                # 算法接口约定等文档
```

## 算法组接入新引擎

系统与算法的唯一接口（见 `extractor.py`）：

```python
class ReIDEngine:
    def __init__(self, weights_path): ...
        # 加载训练好的模型（CLIP-ReID / 掩码引导 / 双分支）
    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        # 输入：BGR 行人小图；输出：L2 归一化特征向量
        return feat
```

交付物：`engine.py`（模型定义）+ `.pth` 权重（放 `weights/`）+ 依赖说明。系统侧换引擎只改 `extractor.py` 一个文件，其余代码不动。换衣双分支模型额外提供 `fuse(feat_a, feat_b) -> feat` 融合函数。

## 待办（中期检查清单）

- [x] 路径集中配置（config.py）
- [x] 引擎统一：search / rank_list / demo 共用 extractor.py 一个引擎入口
- [x] 资源一键下载脚本（scripts/download_assets.py）
- [x] retrieve.py 一键检索（Windows 双击 retrieve.bat 即可运行）
- [x] 三场景生成脚本（scripts/make_scenario.py）
- [x] **Web 原型系统（app.py：视频流接入 + 目标框选 + 跨镜头检索 + Top-K 展示）**
- [x] **视频流演示（stream.py：视频/摄像头/RTSP + ByteTrack 行人编号）**
- [x] **Market-1501 跨镜头演示子集随仓库分发（克隆即可跑）**
- [ ] 遮挡/换衣场景测试集（数据集下载后各跑一次 make_scenario.py）
- [ ] 接入算法组新引擎，新旧引擎对比图

## 三场景测试集生成

```bash
# 常规（Market-1501 官方测试图）
python scripts/make_scenario.py --name 常规 --source data/datasets/Market-1501/bounding_box_test --num 10

# 数据集下载解压到 data/datasets/ 后：
python scripts/make_scenario.py --name 遮挡 --source data/datasets/Occluded-Duke/query --num 10
python scripts/make_scenario.py --name 换衣 --source data/datasets/PRCC/test --num 10
```

每个场景自动生成：`data/scenarios/<场景名>/`（query.jpg + gallery/）和 `outputs/scenario_<场景名>_rank_list.png`。
