import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fast-reid"))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ReIDExtractor:
    def __init__(self, weights_path, device="cpu"):
        self.device = device
        
        # 创建配置
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"
        
        # 手动设置配置
        cfg.MODEL.META_ARCHITECTURE = "Baseline"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.BACKBONE.DEPTH = 50
        cfg.MODEL.HEADS.NAME = "EmbeddingHead"
        cfg.MODEL.HEADS.EMBEDDING_DIM = 2048
        cfg.MODEL.HEADS.NUM_CLASSES = 751
        
        # 构建模型
        self.model = build_model(cfg)
        
        # 加载权重
        state_dict = torch.load(weights_path, map_location='cpu')
        print(f"权重文件中的键数量: {len(state_dict)}")
        if 'model' in state_dict:
            state_dict = state_dict['model']
            print("找到 'model' 键，使用它")
        
        # 加载并检查哪些键被跳过了
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        print(f"缺失的键（模型有但权重没有）: {len(missing_keys)}")
        if missing_keys:
            print(f"  前5个: {missing_keys[:5]}")
        print(f"多余的键（权重有但模型没有）: {len(unexpected_keys)}")
        if unexpected_keys:
            print(f"  前5个: {unexpected_keys[:5]}")
        
        # 检查第一层卷积的权重是否加载成功
        first_conv_weight = list(self.model.parameters())[0]
        print(f"模型第一层权重均值: {first_conv_weight.mean().item():.6f}")
        
        self.model.eval()
        print("✓ FastReID 模型加载完成")
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_np):
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = self.transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(tensor)
            if isinstance(features, tuple):
                features = features[0]
            features = features / torch.norm(features, dim=1, keepdim=True)
            return features.cpu().numpy().flatten()

