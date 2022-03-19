import torch
import torch.nn as nn

__all__ = ["CFPNet"]


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding='same', padding_mode='reflect', activation='ReLU', dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=padding, padding_mode=padding_mode,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.activation = getattr(nn, activation)()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        if self.activation != None:
            output = self.activation(output)
        return output


class FPChannel(nn.Module):
    """
    Feature Pyramid Channel
    """
    def __init__(self, filters, ksize=3, dilation=(1, 1)):
        super().__init__()
        self.filters = filters
        self.x_1 = ConvBN(filters, filters // 4, ksize=ksize, stride=1, dilation=dilation)
        self.x_2 = ConvBN(filters // 4, filters // 4, ksize=ksize, stride=1, dilation=dilation)
        self.x_3 = ConvBN(filters // 4, filters // 2, ksize=ksize, stride=1, dilation=dilation)
        self.bn = nn.BatchNorm2d(filters, eps=1e-3)

    def forward(self, input):
        o_1 = self.x_1(input)
        o_2 = self.x_2(o_1)
        o_3 = self.x_3(o_2)
        cat = torch.cat([o_1, o_2, o_3], 1)
        cat = self.bn(cat)
        return cat


class CFPModule(nn.Module):
    """
    CFP Module for defect segmentation
    """
    def __init__(self, filters, dilation):
        """
        :param filters: number of filters, input and output are the same
        :param dilation: dilation rate
        :param ksize: kernel size
        """
        super().__init__()
        self.x_inp = ConvBN(filters, filters // 4, ksize=3, dilation=dilation)
        self.x_1 = FPChannel(filters // 4, ksize=3, dilation=dilation)
        self.x_2 = FPChannel(filters // 4, ksize=3, dilation=dilation // 4 + 1)
        self.x_3 = FPChannel(filters // 4, ksize=3, dilation=dilation // 2 + 1)
        self.x_4 = FPChannel(filters // 4, ksize=3, dilation=dilation + 1)
        self.bn = nn.BatchNorm2d(filters, eps=1e-3)
        self.conv = ConvBN(filters, filters, ksize=1, dilation=1, stride=1)

    def forward(self, input):

        o_inp = self.x_inp(input)

        o_1 = self.x_1(o_inp)
        o_2 = self.x_2(o_inp)
        o_3 = self.x_3(o_inp)
        o_4 = self.x_4(o_inp)

        add_1 = o_1
        add_2 = add_1 + o_2
        add_3 = add_2 + o_3
        add_4 = add_3 + o_4

        output = torch.cat([add_1, add_2, add_3, add_4], 1)
        output = self.bn(output)
        output  = self.conv(output)
        output = output + input
        
        return output

class DownSampleBy2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.relu = nn.ReLU(in_channels)

    def forward(self, input):
        output = self.avgpool(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class UpSampleBy2(nn.Module):
    """
    Upsample by using `ConvTrans2D(ksize=2, stride=2, activation='relu')`
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.trans_conv(input)
        output = self.relu(output)
        return output


class CFPNetMed(nn.Module):
    def __init__(self, img_channels, return_features=False):
        """
        :param img_channels: number of channels in the input image
        :param return_features: if True, forward() returns (feature, segmentation), 
            dim(feature) = (batch_size, 257, height, width), dim(segmentation) = (batch_size, 1, height, width)
        """
        super().__init__()

        self.img_channels = img_channels
        self.return_features = return_features

        # initial convolution, downsample to 1/2
        self.down_1 = nn.Sequential(
            ConvBN(img_channels, 32, ksize=3, stride=2, dilation=1, padding=1, padding_mode='reflect'),
            ConvBN(32, 32, ksize=3, stride=1, dilation=1),
            ConvBN(32, 32, ksize=3, stride=1, dilation=1),
        )

        # input injection 1/2
        self.inject_1 = DownSampleBy2(img_channels)
        # cat([inject_1, init_conv])
        # downsample to 1/4
        self.down_2 = ConvBN(32 + img_channels, 64, ksize=3, stride=2, dilation=1, padding=1, padding_mode='reflect')

        # CFP block 1
        self.cfp_1 = nn.Sequential(
            CFPModule(64, 2),
            CFPModule(64, 2)
        )

        # input injection 1/4
        self.inject_2 = DownSampleBy2(img_channels)
        # cat([inject_2, cat_1, cfp_1])
        # downsample to 1/8
        self.down_3 = ConvBN(128 + img_channels, 128, ksize=3, stride=2, dilation=1, padding=1, padding_mode='reflect')

        # CFP block 2
        self.cfp_2 = nn.Sequential(
            CFPModule(128, 4),
            CFPModule(128, 4),
            CFPModule(128, 8),
            CFPModule(128, 8),
            CFPModule(128, 16),
            CFPModule(128, 16)
        )

        # input injection 1/8
        self.inject_3 = DownSampleBy2(img_channels)
        # cat([inject_3, cat_2, cfp_2])
       
        # upsampling
        self.deconv_1 = UpSampleBy2(256 + img_channels, 128)    # upsample to 1/4
        # cat([deconv_1, cat_2])
        self.deconv_2 = UpSampleBy2(256 + img_channels, 64)     # upsample to 1/2   
        # cat([deconv_2, cat_1])
        self.deconv_3 = UpSampleBy2(96 + img_channels, 32)      # upsample to 1/1
        
        self.output = nn.Conv2d(32, 1, 1, 1, padding='same')

    def forward(self, input):
        """
        :return: segmentation map (batch_size, 1, height, width)
        :return: features(optional) (batch_size, 257, height, width)
        """

        # initial convolution, downsample to 1/2
        down_1 = self.down_1(input)
        inject_1 = self.inject_1(input)

        # cat input and downsample to 1/4
        cat_1 = torch.cat([inject_1, down_1], 1)
        down_2 = self.down_2(cat_1)
        inject_2 = self.inject_2(inject_1)

        # CFP block 1 and downsample to 1/8
        cfp_1 = self.cfp_1(down_2)
        cat_2 = torch.cat([inject_2, down_2, cfp_1], 1)
        down_3 = self.down_3(cat_2)
        
        # CFP block 2
        cfp_2 = self.cfp_2(down_3)
        inject_3 = self.inject_3(inject_2)
        cat_3 = torch.cat([inject_3, down_3, cfp_2], 1)

        # upsampling
        deconv_1 = self.deconv_1(cat_3)
        up_1 = torch.cat([deconv_1, cat_2], 1)
        deconv_2 = self.deconv_2(up_1)
        up_2 = torch.cat([deconv_2, cat_1], 1)
        deconv_3 = self.deconv_3(up_2)
        output = self.output(deconv_3)

        if self.return_features:
            return output, cat_3
        else:
            return output

