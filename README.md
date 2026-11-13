# SmartGuard-ReID

面向智慧安防的行人重识别（ReID）系统 —— 天津市大学生创新训练计划项目配套原型系统。

支持三种核心场景的行人检索演示：**常规无约束 / 遮挡 / 换衣**。

## 架构

```
视频流/图片 → YOLOv8 行人检测 → ReID 引擎提取特征 → 余弦相似度检索 → Top-K 排名展示
```

- **检测**：`detector.py`（YOLOv8n，ultralytics）
- **特征引擎**：`extractor.py`（当前为 FastReID R50 旧引擎；算法组的 CLIP-ReID 新引擎只需替换此文件内部的模型，接口 `extract(img_bgr) -> 特征向量` 不变）
- **主流程**：`search.py`（检索）+ `rank_list.py`（结果可视化）
- **全局配置**：`config.py`（所有路径集中管理）

## 目录结构

```
SmartGuard-ReID/
├── config.py            # 全局配置（路径、设备）
├── search.py            # 主程序：查询图 → gallery 检索 Top-K
├── detector.py          # YOLOv8 行人检测封装
├── extractor.py         # ReID 特征引擎（唯一换引擎入口）
├── rank_list.py         # 检索结果可视化（横向 rank list）
├── weights/             # 模型权重（YOLO + FastReID R50 + 未来新引擎）
├── fast-reid/           # FastReID 源码（旧引擎依赖）
├── data/
│   ├── query/           # 查询图（CAMPUS HC1 抽帧）
│   ├── gallery/         # 图库（CAMPUS HC2 抽帧）
│   └── scenarios/       # 三场景测试集（常规/遮挡/换衣，待补）
├── outputs/             # 生成的 rank list 等结果
├── scripts/             # 数据准备工具（视频抽帧、行人裁剪）
├── tests/               # 单项测试脚本
├── legacy/              # 历史引擎变体（留档）
├── demo_assets/         # 演示视频
└── docs/                # 算法接口约定等文档
```

## 仓库说明（大文件不入库）

以下内容因体积/版权原因不随仓库分发，克隆后需自行准备：

- **`weights/`**：模型权重。`yolov8n.pt` 从 ultralytics 官方发布下载；`market_bot_R50.pth` 为 FastReID Market-1501 bagtricks R50 预训练权重（约 294MB）
- **`fast-reid/`**：FastReID 源码，来自官方仓库 `github.com/JDAI-CV/fast-reid`，克隆后放入项目根目录
- **`data/`**：查询图/图库图片（CAMPUS 数据集抽帧）及三场景测试图，按各数据集官方渠道获取
- **`outputs/`、`demo_assets/`**：运行生成产物与演示视频

## 快速开始

```bash
# 1. 建环境（语义化命名，不污染 base）
python -m venv reid
reid\Scripts\activate
pip install -r requirements.txt

# 2. 跑检索演示
python search.py
```

## 待办（中期检查清单）

- [ ] demo.py 一键演示脚本
- [ ] 三场景测试集（Market-1501 / Occluded-Duke / PRCC 各一组）
- [ ] Flask 界面（上传 + Top-K 展示）
- [ ] 视频流演示 + ByteTrack 行人编号
- [ ] 接入算法组 CLIP-ReID 新引擎，新旧对比图
