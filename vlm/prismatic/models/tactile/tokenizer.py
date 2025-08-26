import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class TactileEncoder(nn.Module):
    def __init__(self, output_dim=4096, pretrained=True):
        super(TactileEncoder, self).__init__()
        self.output_dim = output_dim
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool
        )
        self.feature_compression = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        if features.dim() == 4:
            features = features.view(features.size(0), features.size(1), -1)
        channel_att = self.attention(features.mean(dim=2))
        channel_att = channel_att.unsqueeze(2)
        attended_features = features * channel_att
        global_feat = attended_features.mean(dim=2)
        encoded = self.feature_compression(global_feat)
        
        return encoded
