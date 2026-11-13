import os
from PIL import Image
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms

# ---------------------- 路径配置（集中在 config.py）----------------------
from config import FASTREID_CONFIG, REID_WEIGHTS, GALLERY_DIR, QUERY_IMAGE, OUTPUT_DIR, DEVICE

# 模型配置与权重
CONFIG_FILE = str(FASTREID_CONFIG)
WEIGHTS_FILE = str(REID_WEIGHTS)

# 数据集路径
GALLERY_DIR = str(GALLERY_DIR)
QUERY_IMG = str(QUERY_IMAGE)

# 展示数量
TOP_K = 10
# -----------------------------------------------------

# 1. 初始化 FastReID 模型
print("✅ 加载 FastReID 模型...")
cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE)
cfg.MODEL.WEIGHTS = WEIGHTS_FILE
# 强制使用 CPU，如果你有 NVIDIA 显卡且配置好了 CUDA，可以改为 "cuda:0"
cfg.MODEL.DEVICE = DEVICE
predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 2. 提取 Gallery 图库特征
print("✅ 提取 Gallery 图库特征...")
gallery_features = []
gallery_paths = []

for img_name in os.listdir(GALLERY_DIR):
    img_path = os.path.join(GALLERY_DIR, img_name)
    if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # 预处理：BGR -> RGB -> PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = predictor(img_tensor)
    gallery_features.append(feat.squeeze().numpy())
    gallery_paths.append(img_path)

gallery_features = np.array(gallery_features)
print(f"✅ 共提取 {len(gallery_features)} 个特征")

# 3. 提取 Query 图片特征
print("✅ 提取 Query 查询图特征...")
query_img = cv2.imread(QUERY_IMG)
if query_img is None:
    print(f"❌ 错误：无法打开查询图 {QUERY_IMG}")
    exit()

query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
query_img_pil = Image.fromarray(query_img)
query_tensor = transform(query_img_pil).unsqueeze(0)

with torch.no_grad():
    query_feat = predictor(query_tensor).squeeze().numpy()


# 4. 计算余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("✅ 计算相似度...")
similarities = []
for feat in gallery_features:
    sim = cosine_similarity(query_feat, feat)
    similarities.append(sim)

# 5. 排序并获取 TopK
sorted_indices = np.argsort(similarities)[::-1]
topk_indices = sorted_indices[:TOP_K]
topk_sims = [similarities[i] for i in topk_indices]
topk_paths = [gallery_paths[i] for i in topk_indices]

# 6. 绘制 Rank List (横向布局版)
print("✅ 生成横向 Rank List 图...")

# 修改点：
# 1. 布局改为 1 行 N+1 列 (1行，1个Query + 10个结果)
# 2. figsize 调整为宽屏长条状 (宽20，高3)
fig, axes = plt.subplots(1, TOP_K + 1, figsize=(20, 3))

# 绘制 Query (放在最左边)
axes[0].imshow(query_img)
axes[0].set_title("Query", fontsize=14, color='blue')
axes[0].axis('off')

# 绘制 Top 10 结果 (依次向右排)
for i in range(TOP_K):
    img = cv2.imread(topk_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 在 axes[i+1] 中画图，因为第0个位置已经被Query占了
    axes[i + 1].imshow(img)

    # 标题显示排名和相似度
    # 相似度大于0.8显示绿色，否则显示黑色 (可选的视觉优化)
    color = 'green' if topk_sims[i] > 0.8 else 'black'
    axes[i + 1].set_title(f"#{i + 1}\n{topk_sims[i]:.3f}", fontsize=12, color=color)
    axes[i + 1].axis('off')

plt.tight_layout()  # 自动调整间距，防止重叠

# 保存结果图
OUTPUT_DIR.mkdir(exist_ok=True)
output_path = str(OUTPUT_DIR / "rank_list_horizontal.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n🎉 完成！横向 Rank List 图已保存为：{output_path}")

# 显示图片
plt.show()
