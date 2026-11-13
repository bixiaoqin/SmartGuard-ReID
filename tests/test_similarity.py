
import cv2
import numpy as np
from detector import PersonDetector
from extractor_resnet import ReIDExtractor

detector = PersonDetector("yolov8n.pt")
extractor = ReIDExtractor()

# 读取两张不同的人的照片
img1 = cv2.imread("person1.jpg")
img2 = cv2.imread("person2.jpg")

# 提取特征
persons1, _ = detector.detect_persons(img1)
persons2, _ = detector.detect_persons(img2)
a = persons1[0]
b = persons2[0]

if len(persons1) > 0 and len(persons2) > 0:
    feat1 = extractor.extract(persons1[0])
    feat2 = extractor.extract(persons2[0])
    
    print(f"\n特征1前10个值: {feat1[:10]}")
    print(f"特征2前10个值: {feat2[:10]}")
    print(f"特征1标准差: {np.std(feat1):.6f}")
    print(f"特征2标准差: {np.std(feat2):.6f}")
    
    # 计算相似度
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    print(f"两张不同人的相似度: {sim:.4f}")
else:
    print("检测失败")