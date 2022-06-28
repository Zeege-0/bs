import torch
import torch.nn as nn
import torchvision

class ResnetOrigin(nn.Module):
    def __init__(self, pretrained=False, num_classes=2):
        super().__init__()
        self.model = torchvision.models.resnet.resnet18(pretrained=pretrained, num_classes=num_classes)
        print("model downloaded")

    def forward(self, x):
        x_expand = x.expand((x.shape[0], 3, x.shape[2], x.shape[3]))
        return self.model(x_expand), torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)

    def set_gradient_multipliers(self, multiplier):
        pass

