import torch
import torchreid
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ReIDExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        # 使用 OSNet 模型，直接在 Market1501 上预训练
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=751,
            pretrained=True
        )
        self.model.eval()
        self.model.to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("✓ OSNet 模型加载完成")

    def extract(self, image_np):
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = self.transform(pil_img).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            features = features / torch.norm(features, dim=1, keepdim=True)
            return features.cpu().numpy().flatten()