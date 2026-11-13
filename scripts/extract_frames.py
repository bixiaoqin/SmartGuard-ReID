import cv2
import os
from detector import PersonDetector

def extract_frames(video_path, output_dir, prefix, max_frames=50):
    """从视频中抽帧保存为图片"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    detector = PersonDetector("yolov8n.pt")
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 30 == 0:
            persons, boxes = detector.detect_persons(frame)
            
            for i, (person, box) in enumerate(zip(persons, boxes)):
                x1, y1, x2, y2 = box
                person_img = frame[y1:y2, x1:x2]
                if person_img.size > 0:
                    filename = f"{prefix}_{saved_count}_{i}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, person_img)
                    print(f"保存: {filename}")
                    saved_count += 1
                    if saved_count >= max_frames:
                        break
        
        frame_count += 1
    
    cap.release()
    print(f"完成！保存了 {saved_count} 张图片到 {output_dir}")

# 从 HC1 抽帧作为 query
extract_frames(
    "D:/database_path/multiview-database/CAMPUS/Garden2/view-HC1.mp4",
    "query",
    "HC1",
    max_frames=30
)

# 从 HC2 抽帧作为 gallery
extract_frames(
    "D:/database_path/multiview-database/CAMPUS/Garden2/view-HC2.mp4",
    "gallery",
    "HC2",
    max_frames=100
)

print("\n✅ 转换完成！")