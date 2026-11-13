import cv2
import os
from ultralytics import YOLO

# ---------------------- 优化后配置 ----------------------
VIDEO_PATH = "D:/my_reid_demo/video.mp4"
SAVE_DIR = "D:/my_reid_demo/person_crops_fixed"  # 新建文件夹，避免覆盖旧数据
CONF_THRESHOLD = 0.6  # 提高置信度，过滤误检小框
FRAME_SKIP = 5  # 隔5帧抽一次，大幅减少重复
MIN_CROP_SIZE = (64, 128)  # 最小有效尺寸，过滤太小的框
TARGET_SIZE = (256, 128)  # ReID标准输入尺寸，统一resize
# ------------------------------------------------------

model = YOLO("yolov8s.pt")  # 换更大的模型，检测更准
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 无法打开视频: {VIDEO_PATH}")
    exit()

frame_count = 0
crop_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    results = model(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)

    for i, det in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        # 过滤太小的框
        if (x2 - x1) < MIN_CROP_SIZE[0] or (y2 - y1) < MIN_CROP_SIZE[1]:
            continue
        # 抠图+统一resize到ReID标准尺寸
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        # 保存
        save_name = f"frame_{frame_count:06d}_crop_{i:03d}.jpg"
        save_path = os.path.join(SAVE_DIR, save_name)
        cv2.imwrite(save_path, crop)
        crop_count += 1

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"✅ 已处理 {frame_count} 帧，已保存 {crop_count} 张有效抠图")

cap.release()
cv2.destroyAllWindows()
print(f"\n🎉 处理完成！共保存 {crop_count} 张高质量行人抠图到 {SAVE_DIR}")
