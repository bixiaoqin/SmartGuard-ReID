# -*- coding: utf-8 -*-
"""
Web 原型系统无头端到端测试：不启动浏览器，用 Flask test_client 模拟全链路
  首页 → 抓帧检测（②）→ 点击框选（③）→ 跨镜头检索（④）→ 结果图 → 上传图片入口

运行（在 reid 环境）：
  python tests/test_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as A  # noqa: E402


def main():
    c = A.app.test_client()

    # 1) 首页
    r = c.get("/")
    assert r.status_code == 200, r.status_code
    html = r.data.decode("utf-8")
    has_video = "view-HC1.mp4" in html
    print(f"[1] 首页 OK，视频选项存在: {has_video}")

    # 2) 抓帧（首次请求会触发引擎 + 图库加载，较慢）
    r = c.post("/frame", json={"video": "view-HC1.mp4", "seek": 30})
    d = r.get_json()
    assert r.status_code == 200, d
    print(f"[2] 抓帧 OK: 第 {d['frame_no']} 帧, {len(d['boxes'])} 个行人, key={d['key']}")

    # 3) 点选第一个行人 → 跨镜头检索
    bbox = d["boxes"][0]["bbox"]
    r = c.post("/select", json={"key": d["key"], "bbox": bbox})
    d2 = r.get_json()
    assert r.status_code == 200, d2
    top1 = d2["results"][0]
    print(f"[3] 跨镜头检索 OK, Top-1: {top1['file']} 相似度 {top1['similarity']}")

    # 4) 结果图可访问
    r = c.get(d2["rank_list"])
    assert r.status_code == 200 and r.mimetype == "image/png"
    print("[4] 结果图访问 OK")

    # 5) 上传图片入口（用默认 Market 查询图测试）
    with open(Path(A.__file__).parent / "data" / "query.jpg", "rb") as f:
        r = c.post("/upload", data={"image": (f, "query.jpg")},
                   content_type="multipart/form-data")
    d3 = r.get_json()
    assert r.status_code == 200, d3
    print(f"[5] 上传图片检索 OK, Top-1 相似度 {d3['results'][0]['similarity']}")

    print("=== Web 原型全链路测试通过 ===")


if __name__ == "__main__":
    main()
