import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ReIDExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        
        # 使用 ResNet50，去掉最后的全连接层
        self.model = models.resnet50(pretrained=True)
        # 去掉分类头，输出2048维特征
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("✓ ResNet50 模型加载完成（ImageNet预训练）")

    def extract(self, image_np):
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = self.transform(pil_img).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            features = features / torch.norm(features, dim=1, keepdim=True)
            return features.cpu().numpy().flatten()