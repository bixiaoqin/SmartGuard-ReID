# -*- coding: utf-8 -*-
"""
安防追踪 Web 原型系统（申报书 ①③④）
  ① 可视化界面：Flask 网页，上传图片 / 监控视频两种入口
  ② 视频流接入：从 CAMPUS 多机位监控视频中逐帧取画面（见 stream.py）
  ③ 目标框选：在标注画面上点击目标行人，后端裁出 query
  ④ 跨镜头检索：query 在其他机位的图库中检索，网页展示 rank list

运行：
  conda activate reid
  python app.py        # 浏览器打开 http://127.0.0.1:5000
"""
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from config import (YOLO_WEIGHTS, REID_WEIGHTS, CAMPUS_VIDEO_DIR,
                    CAMPUS_GALLERY_DIR, GALLERY_DIR, OUTPUT_DIR)
from stream import FrameDetector, grab_annotated_frame
from extractor import ReIDExtractor
from demo import render_rank_list, list_images

app = Flask(__name__)
CACHE_DIR = OUTPUT_DIR / "app_cache"


def pick_gallery_dir():
    """优先用 CAMPUS 跨镜头图库（本地演示）；没有则退回仓库自带的 Market 子集"""
    if CAMPUS_GALLERY_DIR.exists() and any(CAMPUS_GALLERY_DIR.iterdir()):
        return CAMPUS_GALLERY_DIR
    return GALLERY_DIR


class ReIDService:
    """检测器 + 特征引擎 + 图库特征，只加载一次；图库特征缓存到 npz，二次启动秒开"""

    def __init__(self, gallery_dir):
        self.detector = FrameDetector(str(YOLO_WEIGHTS))
        self.engine = ReIDExtractor(str(REID_WEIGHTS))
        self.gallery_dir = Path(gallery_dir)
        self.gallery_paths = []
        self.gallery_feats = None
        self._load_gallery()

    def _load_gallery(self):
        # 缓存名带上父目录，避免 campus/gallery 与 data/gallery 同名冲突
        cache = OUTPUT_DIR / f"gallery_feats_{self.gallery_dir.parent.name}_{self.gallery_dir.name}.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            self.gallery_paths = list(data["paths"])
            self.gallery_feats = data["feats"]
            print(f"[图库] 从缓存加载 {len(self.gallery_paths)} 条特征: {self.gallery_dir}")
            return

        images = list_images(self.gallery_dir)
        feats, paths = [], []
        for i, p in enumerate(images, 1):
            img = cv2.imread(str(p))
            if img is None:
                continue
            try:
                feats.append(self.engine.extract(img))  # 图库图已是裁好的行人小图，直接提特征
            except Exception:
                continue
            paths.append(str(p))
            if i % 20 == 0:
                print(f"[图库] {i}/{len(images)}")
        self.gallery_paths = paths
        self.gallery_feats = np.array(feats)
        OUTPUT_DIR.mkdir(exist_ok=True)
        np.savez(cache, paths=np.array(paths, dtype=object), feats=self.gallery_feats)
        print(f"[图库] 完成 {len(paths)} 条特征，已缓存到 {cache.name}")

    def search(self, query_feat, top_k=5):
        sims = self.gallery_feats @ query_feat / (
            np.linalg.norm(self.gallery_feats, axis=1) * np.linalg.norm(query_feat))
        order = np.argsort(sims)[::-1][:top_k]
        return [{"rank": r + 1, "path": self.gallery_paths[i], "similarity": float(sims[i])}
                for r, i in enumerate(order)]


_service = None


def get_service():
    global _service
    if _service is None:
        _service = ReIDService(pick_gallery_dir())
    return _service


def _run_search(query_crop, key, top_k=5):
    """query 裁图 → 特征 → 检索 → rank list 图，返回响应 JSON"""
    svc = get_service()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CACHE_DIR / f"{key}_query.jpg"), query_crop)
    query_feat = svc.engine.extract(query_crop)
    results = svc.search(query_feat, top_k=top_k)
    out_png = CACHE_DIR / f"{key}_rank.png"
    render_rank_list(query_crop, results, out_png)
    return jsonify(
        rank_list=f"/cache/{out_png.name}",
        results=[{"rank": r["rank"], "similarity": round(r["similarity"], 4),
                  "file": Path(r["path"]).name} for r in results])


def list_videos():
    if not CAMPUS_VIDEO_DIR.exists():
        return []
    return sorted(p.name for p in CAMPUS_VIDEO_DIR.glob("*.mp4"))


@app.route("/")
def index():
    return render_template("index.html", videos=list_videos())


@app.route("/frame", methods=["POST"])
def frame():
    """②视频流接入：从所选视频中抓取一个有行人的帧，返回标注图 + 检测框"""
    data = request.get_json(force=True)
    video_name = data.get("video")
    seek = float(data.get("seek", 0) or 0)
    video_path = CAMPUS_VIDEO_DIR / video_name
    if not video_path.exists():
        return jsonify(error="视频不存在，请检查 config.py 里 CAMPUS_VIDEO_DIR"), 404

    svc = get_service()
    frame_no, frame, dets, annotated = grab_annotated_frame(
        str(video_path), svc.detector, seek_seconds=seek)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{Path(video_name).stem}_{frame_no}_{int(time.time())}"
    cv2.imwrite(str(CACHE_DIR / f"{key}.jpg"), annotated)   # 画框展示图
    cv2.imwrite(str(CACHE_DIR / f"{key}_raw.jpg"), frame)   # 原始帧（框选裁剪用）

    return jsonify(
        key=key,
        frame_no=frame_no,
        image=f"/cache/{key}.jpg",
        boxes=[{"index": i, "bbox": list(d["bbox"]), "conf": round(d["conf"], 2)}
               for i, d in enumerate(dets)])


@app.route("/select", methods=["POST"])
def select():
    """③目标框选 + ④跨镜头检索：点击画面中的行人 → 裁成 query → 到图库检索"""
    data = request.get_json(force=True)
    key = data.get("key")
    bbox = data.get("bbox")
    if not key or not bbox:
        return jsonify(error="缺少 key 或 bbox"), 400
    raw = CACHE_DIR / f"{key}_raw.jpg"
    if not raw.exists():
        return jsonify(error="画面已过期，请重新抓取"), 400
    frame = cv2.imread(str(raw))
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return jsonify(error="裁剪失败"), 400
    return _run_search(crop, key)


@app.route("/upload", methods=["POST"])
def upload():
    """图片上传检索（备用入口）"""
    f = request.files.get("image")
    if not f:
        return jsonify(error="没有收到文件"), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="图片读取失败"), 400
    svc = get_service()
    dets, _ = svc.detector.detect(img)
    if not dets:
        return jsonify(error="图中没有检测到行人"), 400
    crop = svc.detector.crop_person(img, dets[0]["bbox"])
    if crop is None:
        return jsonify(error="行人裁剪失败"), 400
    key = f"upload_{int(time.time())}"
    return _run_search(crop, key)


@app.route("/cache/<path:name>")
def cache_file(name):
    return send_file(CACHE_DIR / name)


if __name__ == "__main__":
    print("=" * 50)
    print("安防追踪原型系统（SmartGuard-ReID）")
    print("浏览器打开 http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=False, host="127.0.0.1", port=5000)
