import os
import cv2
import numpy as np
from glob import glob
from config import YOLO_WEIGHTS, REID_WEIGHTS, GALLERY_DIR, SEARCH_QUERY_IMAGE
from detector import PersonDetector
from extractor import ReIDExtractor

class PersonSearch:
    def __init__(self):
        self.detector = PersonDetector(str(YOLO_WEIGHTS))
        self.extractor = ReIDExtractor(str(REID_WEIGHTS))
    
    def extract_features_from_image(self, image_path, person_index=0):
        """从单张图片中提取特征"""
        persons, boxes = self.detector.detect_persons(image_path)
        if len(persons) == 0:
            print(f"警告: {image_path} 中没有检测到人")
            return None, None
        if person_index >= len(persons):
            print(f"警告: {image_path} 只有 {len(persons)} 个人，索引 {person_index} 超出范围")
            return None, None
        feat = self.extractor.extract(persons[person_index])
        return feat, boxes[person_index]
    
    def build_gallery(self, gallery_folder, person_index=0):
        """
        构建图库：提取文件夹中所有图片的特征
        返回: {
            "image_paths": [...],
            "features": [[...], ...],
            "boxes": [[...], ...]
        }
        """
        # 获取所有图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(gallery_folder, ext)))
            image_paths.extend(glob(os.path.join(gallery_folder, '**', ext), recursive=True))
        
        image_paths = list(set(image_paths))  # 去重
        print(f"找到 {len(image_paths)} 张图片")
        
        features = []
        valid_paths = []
        boxes = []
        
        for i, img_path in enumerate(image_paths):
            print(f"处理中: {i+1}/{len(image_paths)} - {os.path.basename(img_path)}")
            feat, box = self.extract_features_from_image(img_path, person_index)
            if feat is not None:
                features.append(feat)
                valid_paths.append(img_path)
                boxes.append(box)
        
        print(f"成功提取 {len(features)} 个人的特征")
        
        return {
            "image_paths": valid_paths,
            "features": np.array(features),
            "boxes": boxes
        }
    
    def search(self, query_feat, gallery_data, top_k=5):
        """
        在图库中搜索最相似的人
        query_feat: 查询图片的特征
        gallery_data: build_gallery 返回的数据
        top_k: 返回前 k 个最相似的结果
        """
        if len(gallery_data["features"]) == 0:
            print("图库为空")
            return None
        
        # 计算余弦相似度
        similarities = np.dot(gallery_data["features"], query_feat) / (
            np.linalg.norm(gallery_data["features"], axis=1) * np.linalg.norm(query_feat)
        )
        
        # 按相似度降序排序
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 返回 top_k 结果
        results = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            results.append({
                "rank": i + 1,
                "image_path": gallery_data["image_paths"][idx],
                "similarity": float(similarities[idx]),
                "box": gallery_data["boxes"][idx]
            })
        
        return results

def main():
    search_engine = PersonSearch()
    
    # ========== 配置路径（集中在 config.py）==========
    query_image = str(SEARCH_QUERY_IMAGE)   # 查询图片
    query_person_index = 0                  # 查询图片中的第几个人（0=第一个）
    gallery_folder = str(GALLERY_DIR)       # 图库文件夹
    top_k = 5                               # 返回前5个结果
    # ================================
    
    print("=" * 50)
    print("行人检索系统")
    print("=" * 50)
    
    # 1. 提取查询图片特征
    print(f"\n[1/3] 提取查询图片特征: {query_image}")
    query_feat, query_box = search_engine.extract_features_from_image(query_image, query_person_index)
    if query_feat is None:
        print("查询图片中未检测到人，退出")
        return
    print(f"查询图片中人的位置: {query_box}")
    
    # 2. 构建图库
    print(f"\n[2/3] 构建图库: {gallery_folder}")
    if not os.path.exists(gallery_folder):
        print(f"图库文件夹 {gallery_folder} 不存在，正在创建...")
        os.makedirs(gallery_folder)
        print(f"请在 {gallery_folder} 文件夹中放入一些图片，然后重新运行")
        return
    
    gallery_data = search_engine.build_gallery(gallery_folder, person_index=0)
    if len(gallery_data["features"]) == 0:
        print("图库中没有有效的人像，请放入包含行人的图片")
        return
    
    # 3. 搜索
    print(f"\n[3/3] 搜索最相似的 {top_k} 个人...")
    results = search_engine.search(query_feat, gallery_data, top_k)
    
    # 4. 输出结果
    print("\n" + "=" * 50)
    print("检索结果:")
    print("=" * 50)
    for r in results:
        print(f"排名 {r['rank']}: {os.path.basename(r['image_path'])}")
        print(f"  相似度: {r['similarity']:.4f}")
        print(f"  位置: {r['box']}")
    
    # 可选：保存结果图片
    # 在查询图片上画出检索结果（略）

if __name__ == "__main__":
    main()