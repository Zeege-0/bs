import torch
import torch.nn as nn
import torchvision

class DeConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, output_padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.ConvTranspose2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
    

class EfficinetNet(nn.Module):
    def __init__(self, input_channels, b=0, pretrained=False, num_classes=2):
        super().__init__()
        self.model: torchvision.models.EfficientNet = torchvision.models.efficientnet.__dict__[f"efficientnet_b{b}"](pretrained=pretrained, num_classes=num_classes)
        self.deconvs = nn.ModuleList([
            DeConv(112, 40, 3, 2, 1, 1),
            DeConv(80, 24, 3, 2, 1, 1),
            DeConv(48, 16, 3, 2, 1, 1),
            DeConv(32, 32, 3, 2, 1, 1),
            nn.Conv2d(32, input_channels, 3, 1, 1)
        ])
        print("model downloaded")

    def forward(self, x):
        x = x.expand((x.shape[0], 3, x.shape[2], x.shape[3]))
        features = []
        for i in range(9):
            x = self.model.features[i](x)
            if i in [1, 2, 3, 5]:
                features.append(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)

        mask = self.deconvs[0](features[-1])
        mask = self.deconvs[1](torch.cat([mask, features[-2]], 1))
        mask = self.deconvs[2](torch.cat([mask, features[-3]], 1))
        mask = self.deconvs[3](torch.cat([mask, features[-4]], 1))
        mask = self.deconvs[4](mask)

        return x, mask

    def set_gradient_multipliers(self, multiplier):
        pass

if __name__ == '__main__':
    device = 'cuda:0'
    x = torch.rand((1, 3, 224, 224)).to(device)
    model = EfficinetNet(3).to(device)
    y, mask = model(x)
    loss = nn.BCEWithLogitsLoss()(mask, mask)
    loss.backward()
    print(y, y.shape)

