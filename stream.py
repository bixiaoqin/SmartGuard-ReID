# -*- coding: utf-8 -*-
"""
视频流接入（申报书 ②）：把视频文件 / 本地摄像头 / RTSP 流当作监控流处理，
逐帧 YOLO 行人检测 + ByteTrack 编号，为上层界面提供统一的帧迭代器。

用法（命令行自测）：
  python stream.py --video "D:/dataset/database_path/multiview-database/CAMPUS/Garden2/view-HC1.mp4"
  python stream.py --video 0                # 本地摄像头
  python stream.py --video xxx.mp4 --seek 60 --save outputs/stream_sample.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from config import YOLO_WEIGHTS, OUTPUT_DIR


class VideoStream:
    """统一封装视频文件 / 摄像头 / RTSP 的逐帧读取"""

    def __init__(self, source):
        # 纯数字字符串当作摄像头索引，否则当文件/URL
        src = int(source) if str(source).isdigit() else str(source)
        self.source = source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频源: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def seek_seconds(self, seconds):
        """跳到指定秒数（演示时直接翻到有人出现的片段）"""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

    def release(self):
        self.cap.release()


class FrameDetector:
    """帧级行人检测器：返回 (画好框的帧, 检测列表)"""

    def __init__(self, weights_path, use_track=False):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.use_track = use_track  # True 时启用 ByteTrack，给行人分配持续编号

    def detect(self, frame_bgr, conf_threshold=0.5):
        """
        返回 detections: [{bbox:(x1,y1,x2,y2), conf, track_id}, ...]
        以及画好框和编号的帧（供界面展示）
        """
        if self.use_track:
            results = self.model.track(
                frame_bgr, persist=True, conf=conf_threshold,
                classes=[0], verbose=False)[0]
        else:
            results = self.model(frame_bgr, conf=conf_threshold,
                                 classes=[0], verbose=False)[0]

        detections, annotated = [], frame_bgr.copy()
        if results.boxes is None:
            return detections, annotated

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            # ultralytics 的 track 模式会给出 id，普通模式为 None
            tid = int(box.id[0]) if box.id is not None else None
            detections.append({"bbox": (x1, y1, x2, y2), "conf": conf, "track_id": tid})
            color = (36, 168, 93) if tid else (60, 60, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID{tid} {conf:.2f}" if tid else f"{conf:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return detections, annotated

    def crop_person(self, frame_bgr, bbox):
        """按框裁出行人小图（跨镜头检索的 query 就从这里来）"""
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        crop = frame_bgr[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
        return crop if crop.size > 0 else None


def grab_annotated_frame(video_path, detector, seek_seconds=0.0,
                         max_scan_seconds=60, stride=10):
    """
    在视频里找一个"画面里有人"的帧并画框 —— 供 Web 界面快速展示用。
    从 seek_seconds 开始、每隔 stride 帧检测一次，最多扫 max_scan_seconds 秒。
    返回 (帧编号, 帧, detections, annotated)
    """
    stream = VideoStream(video_path)
    if seek_seconds:
        stream.seek_seconds(seek_seconds)
    fps = stream.fps
    max_frames = int(fps * max_scan_seconds)

    found = None
    for i in range(max_frames):
        frame = stream.read()
        if frame is None:
            break
        if i % stride:  # 跳帧采样，CPU 上也能快速扫完
            continue
        detections, annotated = detector.detect(frame)
        if detections:  # 找到有人的帧就返回
            found = (i, frame, detections, annotated)
            break
    stream.release()
    if found is None:
        raise RuntimeError("扫描范围内没有检测到行人，试试把起始秒数往后调")
    return found


def main():
    parser = argparse.ArgumentParser(description="视频流接入自测（申报书 ②）")
    parser.add_argument("--video", required=True, help="视频路径 / 摄像头索引 / RTSP 地址")
    parser.add_argument("--seek", type=float, default=0, help="起始秒数")
    parser.add_argument("--track", action="store_true", help="启用 ByteTrack 行人编号")
    parser.add_argument("--save", default="stream_sample.jpg", help="标注帧保存路径")
    args = parser.parse_args()

    detector = FrameDetector(str(YOLO_WEIGHTS), use_track=args.track)
    frame_no, frame, dets, annotated = grab_annotated_frame(
        args.video, detector, seek_seconds=args.seek)

    print(f"视频源: {args.video}")
    print(f"在第 {frame_no} 帧检测到 {len(dets)} 个行人:")
    for i, d in enumerate(dets, 1):
        tid = f" 编号ID{d['track_id']}" if d["track_id"] is not None else ""
        print(f"  行人{i}: bbox={d['bbox']} 置信度={d['conf']:.2f}{tid}")

    out = OUTPUT_DIR / args.save
    cv2.imwrite(str(out), annotated)
    print(f"标注帧已保存: {out}")


if __name__ == "__main__":
    main()
