# -*- coding: utf-8 -*-
"""
SmartGuard-ReID 全局配置
所有路径、设备参数集中在这里，改路径只改这一个文件。
"""
from pathlib import Path

# 项目根目录（config.py 所在位置）
ROOT = Path(__file__).resolve().parent

# ---------- 模型权重 ----------
WEIGHTS_DIR = ROOT / "weights"
YOLO_WEIGHTS = WEIGHTS_DIR / "yolov8n.pt"          # 行人检测
REID_WEIGHTS = WEIGHTS_DIR / "market_bot_R50.pth"  # 旧引擎 FastReID R50（保留做对比）
# 新引擎权重到位后加一行，例如：
# CLIP_REID_WEIGHTS = WEIGHTS_DIR / "clip_reid.pth"

# ---------- FastReID ----------
FASTREID_ROOT = ROOT / "fast-reid"
FASTREID_CONFIG = FASTREID_ROOT / "configs" / "Market1501" / "bagtricks_R50-ibn.yml"

# ---------- 数据 ----------
DATA_DIR = ROOT / "data"
QUERY_DIR = DATA_DIR / "query"          # 查询图目录
GALLERY_DIR = DATA_DIR / "gallery"      # 默认图库（Market-1501 跨镜头子集，随仓库分发）
QUERY_IMAGE = DATA_DIR / "query.jpg"    # 默认查询图（Market-1501 行人 0001 @ 摄像头c1）
SEARCH_QUERY_IMAGE = DATA_DIR / "query.jpg"  # search.py 默认查询图
SCENARIO_DIR = DATA_DIR / "scenarios"   # 三场景测试集（常规/遮挡/换衣）

# ---------- 完整数据集（本地使用，不入库） ----------
DATASETS_DIR = DATA_DIR / "datasets"
MARKET_DIR = DATASETS_DIR / "Market-1501"
# CAMPUS 多机位监控视频（跨镜头演示的视频源，仅本机存在，不入库）
CAMPUS_VIDEO_DIR = Path("D:/dataset/database_path/multiview-database/CAMPUS/Garden2")
# CAMPUS 其他机位裁好的行人小图（跨镜头检索的图库，不入库）
CAMPUS_GALLERY_DIR = DATA_DIR / "campus" / "gallery"

# ---------- 输出 ----------
OUTPUT_DIR = ROOT / "outputs"           # rank_list 等生成结果

# ---------- 设备 ----------
DEVICE = "cpu"  # 本机无独显用 cpu；实验室/AutoDL 服务器上改 "cuda:0"
