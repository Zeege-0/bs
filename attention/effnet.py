import torch
import torch.nn as nn
import torchvision

class EfficinetNet(nn.Module):
    def __init__(self, input_channels, device, b=0, pretrained=False, num_classes=2):
        super().__init__()
        self.device = device
        self.model: torchvision.models.EfficientNet = torchvision.models.efficientnet.__dict__[f"efficientnet_b{b}"](pretrained=pretrained, num_classes=num_classes)
        self.mask = nn.Conv2d(112, 1, 3, 1, 1)
        print("model downloaded")
        

    def forward(self, x, as_backbone=False):
        if not as_backbone:
            return self.model(x), torch.zeros_like(x, requires_grad=True).to(self.device)
        
        tmp = x
        for i in range(6):
            tmp = self.model.features[i](tmp)
        tmp = self.mask(tmp)
        return self.model(x), tmp

if __name__ == '__main__':
    device = 'cuda:0'
    x = torch.rand((1, 3, 224, 224)).to(device)
    model = EfficinetNet(device).to(device)
    y, mask = model(x)
    loss = nn.BCEWithLogitsLoss()(mask, mask)
    loss.backward()
    print(y, y.shape)

