import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

class ECAAttention(nn.Module):
    """Constructs a ECA module. https://arxiv.org/pdf/1910.03151.pdf
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        k_size = round(np.log2(channel) / gamma + b / gamma)
        k_size = k_size + 1 - k_size % 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)