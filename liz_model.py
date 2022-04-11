import math
import torch
import torch.nn as nn
from attention.eca import ECAAttention
from attention.siman import SimAMAttention
from attention.resnet import BasicBlock, ResLayer
from models import _conv_block, Conv2d_init, FeatureNorm, GradientMultiplyLayer
from CFPNet import CFPNetMed
from CFPNetOrigin import CFPEncoder, Conv


class LizNet(nn.Module):
    def __init__(self, use_med, device, input_width, input_height, input_channels, classes=1):
        """
        :param device: device
        :param input_width: input width
        :param input_height: input height
        :param input_channels: input channels
        :param classes: number of classes
        """
        super(LizNet, self).__init__()
        self.use_med = use_med

        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        if use_med:
            self.volume = CFPNetMed(input_channels, True)
            self.seg_mask = nn.MaxPool2d(kernel_size=8, stride=8)
        else:
            self.volume = CFPEncoder(input_channels)
            self.seg_mask = nn.Sequential(
                Conv2d_init(in_channels=256 + input_channels, out_channels=1, kernel_size=1, padding=0, bias=False),
                SimAMAttention(),
                FeatureNorm(num_features=1, eps=0.001, include_bias=False))
        
        

        self.extractor = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResLayer(BasicBlock, 257 + input_channels, 32, 2, 1),
            ResLayer(BasicBlock, 32, 64, 2, 1),
            ResLayer(BasicBlock, 64, 128, 2, 1),
        )

        # self.extractor = nn.Sequential(nn.MaxPool2d(kernel_size=2),
        #                                _conv_block(in_chanels=64, out_chanels=128, kernel_size=5, padding=2),
        #                                nn.MaxPool2d(kernel_size=2),
        #                                _conv_block(in_chanels=128, out_chanels=256, kernel_size=5, padding=2),
        #                                nn.MaxPool2d(kernel_size=2),
        #                                _conv_block(in_chanels=256, out_chanels=512, kernel_size=5, padding=2))

        self.fc = nn.Sequential(
            nn.Linear(in_features=258, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=classes),
        )

        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply

        self.device = device


    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, x):
        if self.use_med:
            seg_mask, volume = self.volume(x)
            seg_mask = self.seg_mask(seg_mask)
        else:
            volume = self.volume(x) # 28.0
            seg_mask = self.seg_mask(volume) # 28.0

        cat = torch.cat([volume, seg_mask], dim=1) # 28.2

        cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)

        features = self.extractor(cat) # 28.4
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
        global_max_seg = torch.max(torch.max(seg_mask, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg = torch.mean(seg_mask, dim=(-1, -2), keepdim=True)

        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)

        global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
        global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
        global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)

        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
        fc_in = fc_in.reshape(fc_in.size(0), -1)
        prediction = self.fc(fc_in)
        return prediction, seg_mask

