import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork

# [新增] Scale Layer
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

class BackboneWithFPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # 載入預訓練的 ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 我們需要抽取 layer2, layer3, layer4 的特徵
        # ResNet18 的通道數分別為: layer2=128, layer3=256, layer4=512
        self.body = nn.ModuleDict({
            'layer2': backbone.layer2, # Stride 8
            'layer3': backbone.layer3, # Stride 16
            'layer4': backbone.layer4  # Stride 32
        })
        
        # 為了讓資料能流過前面的層 (conv1, bn1, relu, maxpool, layer1)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        
        # 定義 FPN
        # in_channels_list 對應 layer2, layer3, layer4
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512],
            out_channels=out_channels
        )

    def forward(self, x):
        # 通過 ResNet 前段
        x = self.stem(x)
        
        # 抽取特徵
        features = {}
        for key, layer in self.body.items():
            x = layer(x)
            features[key] = x
            
        # 通過 FPN
        # 輸出將是一個字典，包含 'layer2', 'layer3', 'layer4' 對應的增強特徵
        fpn_features = self.fpn(features)
        
        # 將字典轉為 List 輸出 (P3, P4, P5)
        return list(fpn_features.values())
    
class FCOSHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, num_levels=3):
        """
        Args:
            num_levels: FPN 的層數 (預設 3，對應 P3, P4, P5)
        """
        super().__init__()
        
        # 特徵提取層 (Shared convolution towers)
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        )
        
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        )
        
        # 預測層
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # [新增] 為每一層 FPN 定義一個可學習的 Scale
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])
        
        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Focal Loss 的特殊初始化
        nn.init.constant_(self.cls_logits.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)))

    def forward(self, feature_maps):
        logits = []
        bbox_reg = []
        centerness = []
        
        # 遍歷每一層特徵圖 (P3, P4, P5)
        for i, x in enumerate(feature_maps):
            cls_feat = self.cls_tower(x)
            bbox_feat = self.bbox_tower(x)
            
            # Classification
            logits.append(self.cls_logits(cls_feat))
            
            # Centerness
            centerness.append(self.centerness(bbox_feat))
            
            # Regression
            # [修改] 使用 Scale 層： exp(scale(conv(x)))
            reg_pred = self.bbox_pred(bbox_feat)
            
            # 確保 scales 數量足夠 (防止 index out of range)
            if i < len(self.scales):
                reg_pred = self.scales[i](reg_pred)
            
            # 加上 exp 確保距離為正數
            reg_out = torch.exp(reg_pred)
            bbox_reg.append(reg_out)
            
        return logits, bbox_reg, centerness
    
class CorruptionDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # 1. Backbone + FPN
        self.backbone_fpn = BackboneWithFPN(out_channels=256)
        
        # 2. FCOS Head
        # 注意：這裡 num_levels=3 是因為 BackboneWithFPN 回傳 layer2, layer3, layer4 三層
        self.head = FCOSHead(in_channels=256, num_classes=num_classes, num_levels=3)
        
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (Batch, 3, H, W)
        Returns:
            cls_logits: list of tensors
            bbox_regression: list of tensors
            centerness: list of tensors
        """
        # 1. 提取多尺度特徵 (P3, P4, P5)
        features = self.backbone_fpn(images)
        
        # 2. 預測
        cls_logits, bbox_reg, centerness = self.head(features)
        
        return cls_logits, bbox_reg, centerness

if __name__ == "__main__":
    # 簡單測試
    model = CorruptionDetector()
    dummy_img = torch.randn(2, 3, 320, 320)
    cls, reg, ctr = model(dummy_img)
    
    print("Model initialized successfully.")
    print(f"Num scales: {len(reg)}")
    print(f"Scale 0 (P3) shape: {reg[0].shape}")