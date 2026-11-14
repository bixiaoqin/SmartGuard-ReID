# -*- coding: utf-8 -*-
"""
ReID 特征引擎 —— 全系统唯一换引擎入口。

当前引擎：FastReID BoT(R50)（官方 market_bot_R50.pth 权重，Market-1501 Rank-1 94.4%）
算法组新引擎交付后，只需替换本文件内部的模型实现，接口保持不变：
    extract(img_bgr: np.ndarray) -> np.ndarray   # BGR 行人小图 → L2 归一化特征向量
"""
import sys
from pathlib import Path

FASTREID_ROOT = Path(__file__).resolve().parent / "fast-reid"
sys.path.insert(0, str(FASTREID_ROOT))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model


class ReIDExtractor:
    def __init__(self, weights_path, device="cpu"):
        self.device = device

        cfg = get_cfg()
        cfg.merge_from_file(str(Path(__file__).resolve().parent / "fast-reid" /
                                "configs" / "Market1501" / "bagtricks_R50.yml"))
        cfg.MODEL.DEVICE = device
        # 权重是普通 ResNet50（非 IBN 变体），无需再下载 ImageNet 预训练参数
        cfg.MODEL.BACKBONE.DEPTH = "50"      # 本地 yml 基类写了 "50x"(IBN-a)，与权重不符
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.HEADS.NUM_CLASSES = 751    # Market-1501 训练类别数（新版默认 0，需显式指定）
        cfg.freeze()

        self.model = build_model(cfg)

        # 官方 2020 权重与本地 2023 源码的键名差异映射：
        #   heads.bnneck.*     -> heads.bottleneck.0.*
        #   heads.classifier.* -> heads.*
        sd = torch.load(weights_path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        remapped = {}
        for k, v in sd.items():
            if k.startswith("heads.bnneck."):
                k = k.replace("heads.bnneck.", "heads.bottleneck.0.")
            elif k.startswith("heads.classifier."):
                k = k.replace("heads.classifier.", "heads.")
            remapped[k] = v

        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        # pixel_mean/std 是配置常量（权重里冗余），不在属正常；其余缺失/多余应为 0
        real_missing = [k for k in missing if not k.startswith(("pixel_mean", "pixel_std"))]
        real_unexpected = [k for k in unexpected]
        print(f"✓ FastReID BoT(R50) 引擎加载完成（权重键位映射后全部匹配）")
        if real_missing or real_unexpected:
            print(f"  ⚠ 未匹配键 missing={real_missing[:5]} unexpected={real_unexpected[:5]}")

        self.model.eval()

        # 预处理：FastReID 期望 RGB、Resize 到 (256,128)、张量数值范围 [0,255]
        # （模型内部自己做 (x - pixel_mean) / pixel_std，pixel_mean/std 是 0.485*255 这类值）
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),          # → [0,1]
        ])

    def extract(self, image_np):
        rgb = cv2.cvtColor(np.ascontiguousarray(image_np), cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)) * 255.0   # → [0,255]
        with torch.no_grad():
            features = self.model(tensor.unsqueeze(0))
            if isinstance(features, tuple):
                features = features[0]
            features = features / torch.norm(features, dim=1, keepdim=True)
            return features.cpu().numpy().flatten()
