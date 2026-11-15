# -*- coding: utf-8 -*-
"""
一键演示脚本：一条命令跑通完整检索流水线
  选 query → YOLO 检测行人 → ReID 引擎提特征 → gallery 检索 → 输出 rank list 图

用法：
  python retrieve.py                                        # 默认：test1.jpg 搜 data/gallery
  python retrieve.py --query data/query.jpg --gallery data/gallery --topk 10 --name rank_list
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无窗口环境也能出图
import matplotlib.pyplot as plt

from config import QUERY_IMAGE, GALLERY_DIR, OUTPUT_DIR, YOLO_WEIGHTS, REID_WEIGHTS
from detector import PersonDetector
from extractor import ReIDExtractor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DemoPipeline:
    """检测 + 特征引擎只加载一次，重复演示不重复载模型"""

    def __init__(self):
        self.detector = PersonDetector(str(YOLO_WEIGHTS))
        self.engine = ReIDExtractor(str(REID_WEIGHTS))

    def extract_from_image(self, img_bgr, person_index=0):
        """从整幅图（路径/数组均可）提取第 person_index 个行人的特征"""
        persons, boxes = self.detector.detect_persons(img_bgr)
        if not persons:
            raise ValueError("画面中没有检测到行人")
        idx = min(person_index, len(persons) - 1)
        return self.engine.extract(persons[idx]), boxes[idx]


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def run_demo(query_path, gallery_dir, top_k=5, out_name="demo", pipeline=None, verbose=True):
    """执行一次完整检索并输出 rank list 图，返回结果列表"""
    pipeline = pipeline or DemoPipeline()
    query_path = Path(query_path)

    # 1. 查询图特征
    if verbose:
        print(f"[1/3] 提取查询特征: {query_path}")
    query_img = cv2.imread(str(query_path))
    if query_img is None:
        raise FileNotFoundError(f"无法读取查询图 {query_path}")
    query_feat, query_box = pipeline.extract_from_image(query_img)

    # 2. 图库特征
    images = list_images(gallery_dir)
    if not images:
        raise FileNotFoundError(f"图库 {gallery_dir} 中没有图片")
    if verbose:
        print(f"[2/3] 提取图库特征: {gallery_dir}（共 {len(images)} 张）")
    feats, paths = [], []
    for i, p in enumerate(images, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            feat, _ = pipeline.extract_from_image(img)
        except ValueError:
            continue
        feats.append(feat)
        paths.append(p)
        if verbose:
            print(f"  [{i}/{len(images)}] {p.name}")
    if not feats:
        raise RuntimeError("图库中没有可用行人")
    feats = np.array(feats)

    # 3. 余弦相似度检索
    if verbose:
        print(f"[3/3] 检索 Top-{top_k} ...")
    sims = feats @ query_feat / (np.linalg.norm(feats, axis=1) * np.linalg.norm(query_feat))
    order = np.argsort(sims)[::-1][:top_k]
    results = [{"rank": r + 1, "path": paths[i], "similarity": float(sims[i])}
               for r, i in enumerate(order)]

    # 4. 输出
    for r in results:
        print(f"  Rank {r['rank']}: {r['path'].name}  相似度 {r['similarity']:.4f}")

    out_png = OUTPUT_DIR / f"{out_name}_rank_list.png"
    render_rank_list(query_img, results, out_png)
    print(f"完成！rank list 已保存: {out_png}")
    return results


def render_rank_list(query_img, results, out_png):
    """横向 rank list：最左 Query + 依次排列的 Top-K 结果"""
    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 3))

    axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query", fontsize=14, color="blue")
    axes[0].axis("off")

    for i, r in enumerate(results):
        img = cv2.imread(str(r["path"]))
        axes[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color = "green" if r["similarity"] > 0.8 else "black"
        axes[i + 1].set_title(f"#{r['rank']}\n{r['similarity']:.3f}", fontsize=12, color=color)
        axes[i + 1].axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SmartGuard-ReID 一键演示")
    parser.add_argument("--query", default=str(QUERY_IMAGE), help="查询图路径")
    parser.add_argument("--gallery", default=str(GALLERY_DIR), help="图库文件夹")
    parser.add_argument("--topk", type=int, default=5, help="返回前 K 个结果")
    parser.add_argument("--name", default="demo", help="输出文件名前缀")
    parser.add_argument("--person", type=int, default=0, help="查询图中的第几个人（0 起）")
    args = parser.parse_args()

    print("=" * 50)
    print("SmartGuard-ReID 行人检索演示")
    print("=" * 50)
    pipeline = DemoPipeline()
    query_img = cv2.imread(args.query)
    feat, box = pipeline.extract_from_image(query_img, args.person)
    print(f"查询图中行人位置: {box}")
    run_demo(args.query, args.gallery, args.topk, args.name, pipeline)


if __name__ == "__main__":
    main()
