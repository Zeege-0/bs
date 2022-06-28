import torch
import torch.nn as nn
import torchvision


class EfficinetNetOrigin(nn.Module):
    def __init__(self, input_channels, b=0, pretrained=False, num_classes=2):
        super().__init__()
        self.model: torchvision.models.EfficientNet = torchvision.models.efficientnet.__dict__[f"efficientnet_b{b}"](pretrained=pretrained, num_classes=num_classes)
        print("model downloaded")

    def forward(self, x):
        x_expand = x.expand((x.shape[0], 3, x.shape[2], x.shape[3]))
        return self.model(x_expand), torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)

    def set_gradient_multipliers(self, multiplier):
        pass

