import torch
import torch.nn as nn
from CFPNetOrigin import CFPEncoder, InterpolateWrapper, create_model

class CFPNetRealOrigin(nn.Module):
    def __init__(self, img_channels) -> None:
        super().__init__()
        self.model = create_model("Interpolate", img_channels)

    def forward(self, x):
        return torch.zeros((x.shape[0], 2)).to(x.device), self.model(x)[1]
    
    def set_gradient_multipliers(self, multiplier):
        pass

