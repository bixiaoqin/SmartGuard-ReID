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
GALLERY_DIR = DATA_DIR / "gallery"      # 图库目录
QUERY_IMAGE = DATA_DIR / "query.jpg"    # rank_list.py 默认查询图
SEARCH_QUERY_IMAGE = DATA_DIR / "test1.jpg"  # search.py 默认查询图
SCENARIO_DIR = DATA_DIR / "scenarios"   # 三场景测试集（常规/遮挡/换衣）

# ---------- 输出 ----------
OUTPUT_DIR = ROOT / "outputs"           # rank_list 等生成结果

# ---------- 演示 ----------
VIDEO_PATH = ROOT / "demo_assets" / "video.mp4"

# ---------- 设备 ----------
DEVICE = "cpu"  # 本机无独显用 cpu；实验室/AutoDL 服务器上改 "cuda:0"
