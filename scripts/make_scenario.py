# -*- coding: utf-8 -*-
"""
三场景测试集生成脚本：把一组行人小图整理成 场景测试集 + rank list 对比图

适用：下载好数据集后，为 常规/遮挡/换衣 三场景各生成一套 query + gallery + rank list
（这三张 rank list 图就是中期汇报 PPT 的核心素材）

用法示例（下载并解压数据集到 data/datasets/ 后）：
  # 常规场景（Market-1501 测试集）
  python scripts/make_scenario.py --name 常规 --source data/datasets/Market-1501/bounding_box_test --num 10
  # 遮挡场景（Occluded-Duke）
  python scripts/make_scenario.py --name 遮挡 --source data/datasets/Occluded-Duke/query --num 10
  # 换衣场景（PRCC，query 用 test/，gallery 用 query/ 里的换衣图）
  python scripts/make_scenario.py --name 换衣 --source data/datasets/PRCC/test --num 10

可选参数：
  --query <图片路径>   手动指定查询图（默认取源文件夹排序后的第一张）
  --num N              图库图片数量（默认 10）
  --topk K             rank list 显示前 K 名（默认 5）
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SCENARIO_DIR  # noqa: E402
from demo import DemoPipeline, run_demo  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def pick_query_and_gallery(images, num):
    """
    选图策略：优先挑"同一行人出现次数最多"的身份 —— query 用它的一张图，
    图库放它的其他图（跨图命中）+ 其他行人（干扰项）。
    适用于 Market-1501 / Duke 系列的 命名；无法分组时退回顺序取图。
    """
    groups = defaultdict(list)
    for p in images:
        pid = p.name.split("_")[0]
        if pid.isdigit() and pid not in ("-1", "0000"):  # 过滤背景误检/干扰图
            groups[pid].append(p)
    if not groups:
        return images[0], images[1: num + 1]

    pid = max(groups, key=lambda k: len(groups[k]))
    same = groups[pid][1: num // 2 + 1]           # 同一人的其他图（跨镜头/跨时刻）
    others = []
    for k in groups:
        if k != pid:
            others.extend(groups[k])
    return groups[pid][0], (same + others)[:num]


def main():
    parser = argparse.ArgumentParser(description="生成三场景测试集")
    parser.add_argument("--name", required=True, help="场景名（常规/遮挡/换衣）")
    parser.add_argument("--source", required=True, help="源图片文件夹（数据集某个子目录）")
    parser.add_argument("--query", default=None, help="指定查询图路径（默认取第一张）")
    parser.add_argument("--num", type=int, default=10, help="图库图片数量")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    src = Path(args.source)
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if len(images) < 3:
        raise SystemExit(f"源文件夹 {src} 图片太少（{len(images)} 张），至少需要 3 张")

    # 场景目录：data/scenarios/<name>/
    scenario_dir = SCENARIO_DIR / args.name
    gallery_dir = scenario_dir / "gallery"
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(parents=True)

    # 选查询图 + 图库
    if args.query:
        query_src = Path(args.query)
        pool = [p for p in images if p.resolve() != query_src.resolve()]
    else:
        query_src, pool = pick_query_and_gallery(images, args.num)
    picks = pool[: args.num]
    if query_src not in [p for p in images]:
        raise SystemExit(f"指定的查询图不在源文件夹中: {query_src}")

    shutil.copy(query_src, scenario_dir / "query.jpg")
    for i, p in enumerate(picks):
        shutil.copy(p, gallery_dir / f"{i:03d}_{p.name}")

    print(f"场景 [{args.name}] 已生成：")
    print(f"  查询图: {scenario_dir / 'query.jpg'}  （源: {query_src.name}）")
    print(f"  图库:   {gallery_dir}  （{len(picks)} 张）")

    # 生成 rank list 对比图
    run_demo(scenario_dir / "query.jpg", gallery_dir, args.topk,
             out_name=f"scenario_{args.name}", pipeline=DemoPipeline())


if __name__ == "__main__":
    main()
