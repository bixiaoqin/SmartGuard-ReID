# -*- coding: utf-8 -*-
"""
一键下载仓库外的大文件（组员克隆仓库后运行一次即可）：
  1. weights/yolov8n.pt        —— ultralytics 官方发布
  2. weights/market_bot_R50.pth —— 本仓库 Releases（约 294MB）
  3. fast-reid/                —— FastReID 官方源码（git clone）

用法：python scripts/download_assets.py
"""
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
FASTREID_DIR = ROOT / "fast-reid"

# ---------- 下载地址 ----------
# YOLOv8n 检测权重（ultralytics 官方，无需自己上传）
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
# FastReID R50 预训练权重：直接用官方 MODEL_ZOO 发布（BoT(R50)，Market1501，Rank-1 94.4%）
REID_WEIGHT_URL = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth"
# 备用：若官方链接失效或太慢，可换成自己仓库的 Releases 地址（上传同名文件后替换）：
# REID_WEIGHT_URL = "https://github.com/bixiaoqin/SmartGuard-ReID/releases/download/v0.1/market_bot_R50.pth"
# FastReID 源码
FASTREID_REPO = "https://github.com/JDAI-CV/fast-reid.git"


def download(url: str, dest: Path) -> None:
    """带进度地下载文件到指定路径"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[下载] {url}")
    print(f"  -> {dest}")

    def hook(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            pct = done * 100 // total
            sys.stdout.write(f"\r  进度 {pct:3d}%  ({done / 1e6:8.1f} / {total / 1e6:.1f} MB)")
            sys.stdout.flush()

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)
    print(f"[完成] {dest}")


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. YOLO 检测权重
    yolo_dest = WEIGHTS_DIR / "yolov8n.pt"
    if yolo_dest.exists():
        print(f"[跳过] {yolo_dest} 已存在")
    else:
        download(YOLO_URL, yolo_dest)

    # 2. FastReID R50 预训练权重
    reid_dest = WEIGHTS_DIR / "market_bot_R50.pth"
    if reid_dest.exists():
        print(f"[跳过] {reid_dest} 已存在")
    else:
        try:
            download(REID_WEIGHT_URL, reid_dest)
        except Exception as e:
            print(f"\n[失败] ReID 权重下载出错：{e}")
            print("  请确认管理员已创建 v0.1 Release 并上传 market_bot_R50.pth；")
            print("  或手动从 Releases 页面下载后放到 weights/ 目录。")
            sys.exit(1)

    # 3. FastReID 源码
    if FASTREID_DIR.exists():
        print(f"[跳过] {FASTREID_DIR} 已存在")
    else:
        print(f"[克隆] {FASTREID_REPO}")
        r = subprocess.run(["git", "clone", "--depth", "1", FASTREID_REPO, str(FASTREID_DIR)])
        if r.returncode != 0:
            print("[失败] fast-reid 克隆出错，请手动执行：")
            print(f"  git clone --depth 1 {FASTREID_REPO} fast-reid")
            sys.exit(1)

    print("\n全部资源就绪！接下来：")
    print("  1. conda activate reid && pip install -r requirements.txt")
    print("  2. python demo.py   # 验证检索流水线")


if __name__ == "__main__":
    main()
