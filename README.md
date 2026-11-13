# SmartGuard-ReID

面向智慧安防的行人重识别（ReID）系统 —— 天津市大学生创新训练计划项目《面向智慧安防的行人重识别关键算法研究》配套原型系统。

支持三种核心场景的行人检索演示：**常规无约束 / 遮挡 / 换衣**。

## 架构

```
视频流/图片 → YOLOv8 行人检测 → ReID 引擎提取特征 → 余弦相似度检索 → Top-K 排名展示
```

- **检测**：`detector.py`（YOLOv8n，ultralytics）
- **特征引擎**：`extractor.py`（当前为 FastReID R50 旧引擎；算法组的新引擎只需替换此文件内部的模型，接口 `extract(img_bgr) -> 特征向量` 不变）
- **主流程**：`search.py`（检索）+ `rank_list.py`（结果可视化）
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
| `market_bot_R50.pth`（ReID 旧引擎权重，约 294MB） | 本仓库 Releases 页 | `weights/` |
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
python search.py
```

输出 Top-5 检索结果即环境搭建成功。

## 数据集下载（算法组训练 / 三场景测试用）

| 场景 | 数据集 | 下载地址 | 放置位置 |
|---|---|---|---|
| 常规 | Market-1501 | Kaggle：`kaggle.com/datasets/pengcw1/market-1501` | `data/datasets/Market-1501/` |
| 遮挡 | Occluded-Duke | `github.com/lightas/Occluded-DukeMTMC-Dataset` | `data/datasets/Occluded-Duke/` |
| 换衣 | PRCC | 官方申请页：`www.isee-ai.cn/~yangqize/clothing.html` | `data/datasets/PRCC/` |
| 跨镜头视频演示 | CAMPUS（组内已有） | 联系管理员拷贝 | 任意位置，用 `scripts/extract_frames.py` 抽帧 |

**下载后怎么做：**

1. **Market-1501**：解压后应得到 `bounding_box_train/`、`bounding_box_test/`、`query/` 等文件夹，整个数据集文件夹放进 `data/datasets/Market-1501/`
2. **Occluded-Duke**：此仓库**只提供文件名列表，不含图片**——先下载原始 DukeMTMC-reID（Kaggle：`kaggle.com/datasets/igorkrashenyi/dukemtmc-reid`），再用其仓库里的 `convert_duke_to_occduke.py` 脚本转换，生成的 `bounding_box_train/`、`bounding_box_test/`、`query/` 放进 `data/datasets/Occluded-Duke/`
3. **PRCC**：在官方页面提交申请（通常通过百度网盘/邮件发放），解压后的 `prcc/`（含 `train/`、`test/`、`query/`）放进 `data/datasets/PRCC/`
4. 数据集只放本地，**不要提交到 git**（`.gitignore` 已排除 `data/`，防止误传版权数据）

系统演示（`search.py`）不依赖这些数据集，它用的是 `data/query/` + `data/gallery/`（CAMPUS 视频抽帧，仓库已含）；数据集主要供算法组训练评测和挑选三场景测试图。

## 目录结构

```
SmartGuard-ReID/
├── config.py            # 全局配置（路径、设备）
├── search.py            # 主程序：查询图 → gallery 检索 Top-K
├── detector.py          # YOLOv8 行人检测封装
├── extractor.py         # ReID 特征引擎（唯一换引擎入口）
├── rank_list.py         # 检索结果可视化（横向 rank list）
├── weights/             # 模型权重（脚本下载，不入库）
├── fast-reid/           # FastReID 源码（脚本克隆，不入库）
├── data/
│   ├── query/           # 查询图（CAMPUS HC1 抽帧，已含）
│   ├── gallery/         # 图库（CAMPUS HC2 抽帧，已含）
│   └── scenarios/       # 三场景测试集（常规/遮挡/换衣，待补）
├── outputs/             # 生成的 rank list 等结果
├── scripts/             # 资源下载、视频抽帧、行人裁剪工具
├── tests/               # 单项测试脚本
├── legacy/              # 历史引擎变体（留档）
├── demo_assets/         # 演示视频
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
- [x] 资源一键下载脚本
- [ ] demo.py 一键演示脚本
- [ ] 三场景测试集（Market-1501 / Occluded-Duke / PRCC 各一组）
- [ ] Flask 界面（上传 + Top-K 展示）
- [ ] 视频流演示 + ByteTrack 行人编号
- [ ] 接入算法组新引擎，新旧引擎对比图
