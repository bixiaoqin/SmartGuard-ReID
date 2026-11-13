# -*- coding: utf-8 -*-
"""
旧入口（保留兼容）：从 data/gallery 中检索 data/query.jpg 最相似的 Top-10 并输出 rank list。
与 demo.py 共用同一引擎实现（extractor.py），不再单独走 FastReID DefaultPredictor 配置。

完整功能请用 demo.py：
  python demo.py --query data/query.jpg --gallery data/gallery --topk 10 --name rank_list
"""
from config import QUERY_IMAGE, GALLERY_DIR
from demo import run_demo

if __name__ == "__main__":
    run_demo(query_path=QUERY_IMAGE, gallery_dir=GALLERY_DIR,
             top_k=10, out_name="rank_list")
