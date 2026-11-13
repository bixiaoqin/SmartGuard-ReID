from ultralytics import YOLO
import cv2
import os

# 加载模型
model = YOLO("yolov8n.pt")

# 读取图片
img_path = "test.jpg"
img = cv2.imread(img_path)
if img is None:
    print("图片读取失败！")
    exit()

# 检测
results = model(img)

# 创建保存裁剪图的文件夹
save_dir = "person_crops"
os.makedirs(save_dir, exist_ok=True)

# 过滤并保存人的检测框（只保留置信度高的人）
person_count = 0
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # 只保留置信度>0.5的人（过滤掉低置信度的误检）
        if cls == 0 and conf > 0.5:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 裁剪
            crop = img[y1:y2, x1:x2]
            # 保存
            crop_path = os.path.join(save_dir, f"person_{person_count}.jpg")
            cv2.imwrite(crop_path, crop)
            print(f"已保存第{person_count}个人的裁剪图: {crop_path}")

print(f"共检测到{person_count}个有效行人，裁剪图已保存在person_crops文件夹")