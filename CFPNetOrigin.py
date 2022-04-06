import torch
import torch.nn as nn
import torch.nn.functional as F
from van import Attention, Block

__all__ = ["CFPNet"]


class MultiInputSequential(nn.Sequential):
    def forward(self, x, *args):
        for module in self._modules.values():
            x = module(x, *args)
        return x


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
    
    

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output



class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()
        
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)
        
        self.dconv3x1_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(1*d+1, 0), dilation=(d+1,1), groups = nIn //16, bn_acti=True)
        self.dconv1x3_4_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1*d+1), dilation=(1,d+1), groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(1*d+1, 0), dilation=(d+1,1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_4_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1*d+1), dilation=(1,d+1),groups = nIn //16, bn_acti=True)        
        
        self.dconv3x1_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(1*d+1, 0), dilation=(d+1,1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_4_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, 1*d+1), dilation=(1,d+1),groups = nIn //8, bn_acti=True) 
        
        self.dconv3x1_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(1, 0),groups = nIn //16, bn_acti=True)
        self.dconv1x3_1_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1),groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(1, 0),groups = nIn //16, bn_acti=True)
        self.dconv1x3_1_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, 1),groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(1, 0),groups = nIn //16, bn_acti=True)
        self.dconv1x3_1_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, 1),groups = nIn //8, bn_acti=True)
        
        
        self.dconv3x1_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/4+1), 0), dilation=(int(d/4+1),1), groups = nIn //16, bn_acti=True)
        self.dconv1x3_2_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/4+1)), dilation=(1,int(d/4+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/4+1), 0), dilation=(int(d/4+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_2_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/4+1)), dilation=(1,int(d/4+1)),groups = nIn //16, bn_acti=True)        
        
        self.dconv3x1_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(int(d/4+1), 0), dilation=(int(d/4+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_2_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, int(d/4+1)), dilation=(1,int(d/4+1)),groups = nIn //8, bn_acti=True)         
        
        
        
        self.dconv3x1_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/2+1), 0), dilation=(int(d/2+1),1), groups = nIn //16, bn_acti=True)
        self.dconv1x3_3_1 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/2+1)), dilation=(1,int(d/2+1)), groups = nIn //16, bn_acti=True)
        
        self.dconv3x1_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, 1), 1,
                              padding=(int(d/2+1), 0), dilation=(int(d/2+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_3_2 = Conv(nIn // 16, nIn // 16, (1, dkSize), 1,
                              padding=(0, int(d/2+1)), dilation=(1,int(d/2+1)),groups = nIn //16, bn_acti=True)        
        
        self.dconv3x1_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, 1), 1,
                              padding=(int(d/2+1), 0), dilation=(int(d/2+1),1),groups = nIn //16, bn_acti=True)
        self.dconv1x3_3_3 = Conv(nIn // 8, nIn // 8, (1, dkSize), 1,
                              padding=(0, int(d/2+1)), dilation=(1,int(d/2+1)),groups = nIn //8, bn_acti=True)              
        
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0,bn_acti=False)
        
    def forward(self, input, streams):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)
        
        with torch.cuda.stream(streams[0]):
            o1_1 = self.dconv3x1_1_1(inp)
            o1_1 = self.dconv1x3_1_1(o1_1)
            o1_2 = self.dconv3x1_1_2(o1_1)
            o1_2 = self.dconv1x3_1_2(o1_2)
            o1_3 = self.dconv3x1_1_3(o1_2)
            o1_3 = self.dconv1x3_1_3(o1_3)
            output_1 = torch.cat([o1_1,o1_2,o1_3], 1)

            o2_1 = self.dconv3x1_2_1(inp)
            o2_1 = self.dconv1x3_2_1(o2_1)
            o2_2 = self.dconv3x1_2_2(o2_1)
            o2_2 = self.dconv1x3_2_2(o2_2)
            o2_3 = self.dconv3x1_2_3(o2_2)
            o2_3 = self.dconv1x3_2_3(o2_3) 
            output_2 = torch.cat([o2_1,o2_2,o2_3], 1) 
    
        with torch.cuda.stream(streams[1]):
            o3_1 = self.dconv3x1_3_1(inp)
            o3_1 = self.dconv1x3_3_1(o3_1)
            o3_2 = self.dconv3x1_3_2(o3_1)
            o3_2 = self.dconv1x3_3_2(o3_2)
            o3_3 = self.dconv3x1_3_3(o3_2)
            o3_3 = self.dconv1x3_3_3(o3_3)
            output_3 = torch.cat([o3_1,o3_2,o3_3], 1)          
        
            o4_1 = self.dconv3x1_4_1(inp)
            o4_1 = self.dconv1x3_4_1(o4_1)
            o4_2 = self.dconv3x1_4_2(o4_1)
            o4_2 = self.dconv1x3_4_2(o4_2)
            o4_3 = self.dconv3x1_4_3(o4_2)
            o4_3 = self.dconv1x3_4_3(o4_3)
            output_4 = torch.cat([o4_1,o4_2,o4_3], 1)  
        
        torch.cuda.current_stream().wait_stream(streams[0])
        torch.cuda.current_stream().wait_stream(streams[1])

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4],1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        
        return output + input
        

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class CFPEncoder(nn.Module):
    def __init__(self, img_channels, block_1=2, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(img_channels, 32, 3, 2, padding=3, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + img_channels)

        # Attention Block 1
        self.att1 = nn.Sequential(
            Conv(64, 64, 5, 1, padding=2, bn_acti=True),
            Conv(64, 64, 5, 1, padding=2, bn_acti=True),
            Conv(64, 64, 5, 1, padding=2, bn_acti=True),
        )

        # Attention Block 2
        self.att2 = nn.Sequential(
            Conv(192, 192, 5, 1, padding=2, bn_acti=True),
            Conv(192, 192, 5, 1, padding=2, bn_acti=True),
            Conv(192, 192, 5, 1, padding=2, bn_acti=True),
            Attention(192)
        )
        
        # CFP Block 1
        dilation_block_1 =[1, 3]
        self.downsample_1 = DownSamplingBlock(32 + img_channels, 64)
        self.CFP_Block_1 = MultiInputSequential()
        for i in range(0, block_1):
            self.CFP_Block_1.add_module("CFP_Module_1_" + str(i), CFPModule(64, d=dilation_block_1[i]))
            
        self.bn_prelu_2 = BNPReLU(192 + img_channels)

        # CFP Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16] #camvid #cityscapes [4,4,8,8,16,16] # [4,8,16]
        self.downsample_2 = DownSamplingBlock(192 + img_channels, 192)
        self.CFP_Block_2 = MultiInputSequential()
        for i in range(0, block_2):
            self.CFP_Block_2.add_module("CFP_Module_2_" + str(i),
                                        CFPModule(192, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(576 + img_channels)

    def forward(self, x, streams):

        o2x32 = self.init_conv(x)
        down_1 = self.down_1(x)
        down_2 = self.down_2(x)
        down_3 = self.down_3(x)
        o2x32_cat = self.bn_prelu_1(torch.cat([o2x32, down_1], 1))

        # Stage 1
        o4x64 = self.downsample_1(o2x32_cat)
        # Attention Block 1
        with torch.cuda.stream(streams[0]):
            att1 = self.att1(o4x64)
        # CFP Block 1
        cfp1 = self.CFP_Block_1(o4x64, streams[1:])
        torch.cuda.synchronize()
        o4x128 = self.bn_prelu_2(torch.cat([att1, cfp1, o4x64, down_2], 1))

        # Stage 2
        o8x128 = self.downsample_2(o4x128)
        # Attention Block 2
        with torch.cuda.stream(streams[0]):
            att2 = self.att2(o8x128)
        # CFP Block 2
        cfp2 = self.CFP_Block_2(o8x128, streams[1:])
        torch.cuda.synchronize()
        o8x256 = self.bn_prelu_3(torch.cat([att2, cfp2, o8x128, down_3], 1))

        return o8x256
    
