import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor

from model import resnet, efficientnetv2, torch_efficientnet
import Attentions

resnet_dict = {
    "resnet18": resnet.resnet18,
    "resnet34": resnet.resnet34,
    "resnet50": resnet.resnet50,
    "resnet101": resnet.resnet101,
    "resnet152": resnet.resnet152,
}

efficientnetv2_dict = {
    'efficientnet_v2s': efficientnetv2.efficientnetv2_s,
    'efficientnet_v2m': efficientnetv2.efficientnetv2_m,
    'efficientnet_v2l': efficientnetv2.efficientnetv2_l,
}

torch_efficientnetv2dict = {
    'torch_efficientnet_v2s': torch_efficientnet.efficientnet_v2_s,
    'torch_efficientnet_v2m': torch_efficientnet.efficientnet_v2_m,
    'torch_efficientnet_v2l': torch_efficientnet.efficientnet_v2_l,
}

### For latest triplet_attention module code please refer to the corresponding file in root.
def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif 'torch_efficientnet' in name.lower():
        return torch_EfficientNetBackbone(name)
    elif 'efficientnet' in name.lower():
        return EfficientNetBackbone(name)

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim

class torch_EfficientNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(torch_EfficientNetBackbone, self).__init__()
        efficientnetv2 = torch_efficientnetv2dict[network_type](pretrained=False)
        self.features = efficientnetv2.features
        self.CAM = Attentions.CoordAtt(efficientnetv2.classifier[-1].in_features, efficientnetv2.classifier[-1].in_features) # CAM
        # self.CAB = Attentions.Category_attention(efficientnetv2.classifier[-1].in_features, 5) # CA
        self.avgpool = efficientnetv2.avgpool
        self._feature_dim = efficientnetv2.classifier[-1].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.CAM(x)
        # x = self.CAB(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

class EfficientNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(EfficientNetBackbone, self).__init__()
        efficientnet1 = efficientnetv2_dict[network_type](pretrained=False)
        self.stem = efficientnet1.stem
        self.blocks = efficientnet1.blocks
        self.head = efficientnet1.head[:1]
        # self.CAM = Attentions.CoordAtt(efficientnet1.head[-1].in_features, efficientnet1.head[-1].in_features) # CAM
        # self.CAB = Attentions.Category_attention(efficientnet1.head[-1].in_features, 5) # CA
        self.avgpool = efficientnet1.head[1]
        self._feature_dim = efficientnet1.head[-1].in_features
        del efficientnet1

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        # x = self.CAM(x)
        # x = self.CAB(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim
