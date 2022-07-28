import torch.nn as nn
import torch
from network.nn.operators import AlignedModule, PSPModule

from network import resnet_d as Resnet_Deep
from network.nn.mynn import Norm2d, Upsample
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class SideOutput(nn.Module):
    def __init__(self, side_in_channels):
        super(SideOutput, self).__init__()
        self.side_output = nn.ModuleList()

        for i in range(len(side_in_channels)):
            self.side_output.append(nn.Sequential(
                nn.Conv2d(side_in_channels[i], 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU()
            ))

        self.fuse = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: p2-p5
        h, w = x[0].shape[2:]
        side_outputs = []
        for i in range(len(x) - 1, 0, -1):
            side_outputs.append(
                nn.functional.interpolate(self.side_output[i](x[i]), (h, w), mode='bilinear', align_corners=True))
        side_outputs.append(self.side_output[0](x[0]))
        output = self.fuse(torch.cat(side_outputs, dim=1))
        return self.sigmoid(output)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6), norm_layer=None):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + len(bins) * out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = self.fusion(torch.cat(out, 1))
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample=None, norm_layer=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.down_sample is not None:
            residual = self.down_sample(x)
        out = self.relu(out + residual)
        return out


class HighPassFilter(nn.Module):
    def __init__(self, channels, kernel_size):
        super(HighPassFilter, self).__init__()
        "here we implement average filter"
        self.kernel_size = kernel_size
        kernel = torch.ones(kernel_size, dtype=torch.float)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        # self.register_buffer('weight', kernel)
        self.groups = channels
        self.conv = nn.functional.conv2d

    def _computer_padding(self, kernel_size):
        # padding, padH, padW. Suppose k is odd. Else, pad_l = (k-1) // 2, pad_r = (k-1) - pad_l
        padding = [(k - 1) // 2 for k in kernel_size]
        return tuple(padding)

    def forward(self, x):
        padding = self._computer_padding(self.kernel_size)
        blur = self.conv(x, weight=self.weight, padding=padding, groups=self.groups)
        high_pass_filtered = x - blur
        return high_pass_filtered


class StructureEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=None):
        super(StructureEnhance, self).__init__()
        self._high_pass_filter = HighPassFilter(in_channels, kernel_size)
        # here maybe only the conv, no bn_relu.
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, stride=1, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res = ResBlock(out_channels, out_channels, stride=1, norm_layer=norm_layer)
        self.bn_relu = nn.Sequential(
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # high pass filter -> cat(low,high) -> resnet -> output + high
        # low level, high level
        low_feat, high_feat = x
        l_h, l_w = low_feat.shape[2:]
        up_original_high_feat = F.interpolate(high_feat, size=(l_h, l_w), mode='bilinear', align_corners=True)

        # high pass filter
        low_feat = self._high_pass_filter(low_feat)
        high_feat = self._high_pass_filter(high_feat)
        # cat high,low feature
        up_high_feat = F.interpolate(high_feat, size=(l_h, l_w), mode='bilinear', align_corners=True)
        mix_feature = torch.cat([low_feat, up_high_feat], 1)
        mix_feature = self.conv_bn_relu(mix_feature)
        mix_feature = self.res(mix_feature)
        mix_feature = mix_feature + up_original_high_feat
        mix_feature = self.fusion(self.bn_relu(mix_feature))
        return mix_feature


class DetailEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=None):
        super(DetailEnhance, self).__init__()
        self._high_pass_filter = HighPassFilter(in_channels, kernel_size)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, stride=1, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res = ResBlock(out_channels, out_channels, stride=1, norm_layer=norm_layer)
        self.bn_relu = nn.Sequential(
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        low_feat, high_feat = x
        h_h, h_w = high_feat.shape[2:]
        original_high_feat = high_feat
        # high pass filter
        low_feat = self._high_pass_filter(low_feat)
        high_feat = self._high_pass_filter(high_feat)
        # cat high,low feature
        down_low_feat = F.interpolate(low_feat, size=(h_h, h_w), mode='bilinear', align_corners=True)
        mix_feature = torch.cat([high_feat, down_low_feat], 1)
        mix_feature = self.conv_bn_relu(mix_feature)
        mix_feature = self.res(mix_feature)
        mix_feature = mix_feature + original_high_feat
        mix_feature = self.fusion(self.bn_relu(mix_feature))
        return mix_feature


class DenseConnect(nn.Module):
    def __init__(self):
        super(DenseConnect, self).__init__()

    def forward(self, x):
        # p2-p5
        for i in range(len(x) - 2, -1, -1):
            for j in range(i + 1, len(x)):
                x[i] = x[i] + F.interpolate(x[j], size=x[i].shape[2:], mode='bilinear', align_corners=True)

        return x


class FEModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(9, 9), norm_layer=None):
        super(FEModule, self).__init__()
        self.in_channels = in_channels
        self.conv_reduction = nn.ModuleList()
        self.se = nn.ModuleList()
        self.de = nn.ModuleList()

        for i in range(len(in_channels)):
            if i is not len(in_channels) - 1:
                self.conv_reduction.append(nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, 1, padding=0, bias=False),
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True)
                ))

        self.conv_reduction.append(PPM(in_channels[-1], out_channels, norm_layer=norm_layer))

        for i in range(len(in_channels) - 1):
            self.se.append(nn.Sequential(
                StructureEnhance(out_channels, out_channels, kernel_size, norm_layer=norm_layer)
            ))
            self.de.append(nn.Sequential(
                DetailEnhance(out_channels, out_channels, kernel_size, norm_layer=norm_layer)
            ))

    def forward(self, x):
        reductions = [
            reduction_conv(x[i])
            for i, reduction_conv in enumerate(self.conv_reduction)
        ]

        for i in range(len(self.in_channels) - 1):
            reductions[i + 1] = self.de[i]([reductions[i], reductions[i + 1]])
        for i in range(len(self.in_channels) - 1, 0, -1):
            reductions[i - 1] = self.se[i - 1]([reductions[i - 1], reductions[i]])

        return reductions


class FENet(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D', fpn_dsn=False, boundary=False,
                 dense_connected=False):
        super(FENet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.fpn_dsn = fpn_dsn
        self.boundary = boundary
        self.dense_connected = dense_connected

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
        elif trunk == 'resnet-18-deep':
            resnet = Resnet_Deep.resnet18()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        in_channels = [64, 128, 256, 512]
        side_in_channels = [64, 128]
        out_channels = 128
        if trunk == 'resnet-18-deep':
            self.fpn_fe = FEModule(in_channels=in_channels, out_channels=out_channels, kernel_size=(9, 9),
                                   norm_layer=Norm2d)
        else:
            in_channels = [256, 512, 1024, 2048]
            side_in_channels = [256, 512]
            out_channels = 256
            self.fpn_fe = FEModule(in_channels=in_channels, out_channels=out_channels, kernel_size=(9, 9),
                                   norm_layer=Norm2d)

        self.conv_last = nn.Sequential(
            nn.Conv2d(len(in_channels) * out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            Norm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )

        if self.dense_connected:
            self.dense = DenseConnect()

        if self.training and self.fpn_dsn:
            self.dsn = nn.ModuleList()
            for i in range(2, len(in_channels)):
                self.dsn.append(nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, kernel_size=3, stride=1, padding=1),
                    Norm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                ))
        if self.training and self.boundary:
            self.side_output = SideOutput(side_in_channels=side_in_channels)

    def forward(self, x, gts=None):
        dsn_out = []
        side_boundary = []
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        if self.fpn_dsn and self.training:
            dsn_out.append(self.dsn[0](x3))
            dsn_out.append(self.dsn[1](x4))
        if self.boundary and self.training:
            # 四层或者两层
            side_boundary.append(self.side_output([x1, x2]))
        fpn_feature_list = self.fpn_fe([x1, x2, x3, x4])
        if self.dense_connected:
            fpn_feature_list = self.dense(fpn_feature_list)

        fusion_list = [fpn_feature_list[0]]
        output_size = fpn_feature_list[0].size()[2:]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i], output_size, mode='bilinear', align_corners=True
            ))
        out = self.conv_last(torch.cat(fusion_list, dim=1))
        main_out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            # dsn_out = []
            # if self.fpn_dsn:
            #     for i in range(len(fpn_feature_list)):
            #         # p2-p5
            #         dsn_out.append(self.dsn[i](fpn_feature_list[i]))
            return self.criterion([out, dsn_out, side_boundary], gts)
        return main_out


def DeepR18_Baseline(num_classes, criterion):
    """
    Baseline
    """
    pass


def DeepR18_FE_deeply(num_classes, criterion):
    """
    ResNet-18 Based Network-FENet
    """
    return FENet(num_classes, trunk='resnet-18-deep', criterion=criterion, variant='D')


def DeepR18_FE_deeply_completely(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return FENet(num_classes, trunk='resnet-18-deep', criterion=criterion, variant='D', fpn_dsn=True, boundary=False,
                 dense_connected=False)


def DeepR50_FE_deeply_completely(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return FENet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', fpn_dsn=True, boundary=False,
                 dense_connected=False)


def DeepR101_FE_deeply_completely(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return FENet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', fpn_dsn=True, boundary=False,
                 dense_connected=False)
