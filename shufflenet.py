import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
from octconv import OctConv2d

class _BatchNorm2d(nn.Module):
  def __init__(self, num_features, alpha_in=0.5, alpha_out=0.5, eps=1e-5, momentum=0.1, affine=True,
               track_running_stats=True):
    super(_BatchNorm2d, self).__init__()
    hf_ch = int(num_features * (1 - alpha_out))
    lf_ch = num_features - hf_ch
    self.bnh = nn.BatchNorm2d(hf_ch)
    self.bnl = nn.BatchNorm2d(lf_ch)
  def forward(self, x):
    hf, lf = x
    return self.bnh(hf), self.bnl(lf)

class _Cat(nn.Module):
  def forward(self, x, dim=0):
    hf1, lf1 = x[0]
    hf2, lf2 = x[1]
    return torch.cat((hf1, hf2), dim), torch.cat((lf1, lf2), dim)

def _avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    #hf, lf = input
    #CHANGE TO F.interpolate()
    #hf = F.avg_pool2d(hf, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
    #lf = F.avg_pool2d(lf, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
    return _downsample2x(input)

def _downsample2x(input):
    hf, lf = input
    new_hf_dim = int(hf.shape[2]/2), int(hf.shape[3]/2)
    new_lf_dim = int(lf.shape[2]/2), int(lf.shape[3]/2)
    hf = F.interpolate(hf, new_hf_dim, mode='bilinear', align_corners=True)
    lf = F.interpolate(lf, new_lf_dim, mode='bilinear', align_corners=True)
    return hf, lf

class _ReLU(nn.Module):
  def __init__(self, inplace=False):
    super(_ReLU, self).__init__()
    self.relu = nn.ReLU(inplace=inplace)
  def forward(self, x):
    hf, lf = x
    return self.relu(hf), self.relu(lf)

class _MaxPool2d(nn.Module):
  def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    super(_MaxPool2d, self).__init__()
    self.maxpool = nn.MaxPool2d(kernel_size, 
                    stride=stride, 
                    padding=padding, 
                    dilation=dilation, 
                    return_indices=return_indices, ceil_mode=ceil_mode)
  def forward(self, x):
    hf, lf = x
    hf = self.maxpool(hf)
    lf = self.maxpool(lf)
    return hf, lf

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1, alpha=0.5):    
    """3x3 convolution with padding
    """
    return OctConv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
        alpha=alpha)


def conv1x1(in_channels, out_channels, groups=1, alpha=0.5):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return OctConv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        groups=groups,
        stride=1,
        alpha=alpha)


def channel_shuffle(x, groups):
    hf, lf = x
    batchsize_hf, num_channels_hf, height_hf, width_hf = hf.data.size()
    batchsize_lf, num_channels_lf, height_lf, width_lf = lf.data.size()

    channels_per_group_hf = num_channels_hf // groups
    channels_per_group_lf = num_channels_lf // groups
    
    # reshape
    hf = hf.view(batchsize_hf, groups, 
        channels_per_group_hf, height_hf, width_hf)
    lf = lf.view(batchsize_lf, groups, 
        channels_per_group_lf, height_lf, width_lf)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    hf = torch.transpose(hf, 1, 2).contiguous()
    lf = torch.transpose(lf, 1, 2).contiguous()

    # flatten
    hf = hf.view(batchsize_hf, -1, height_hf, width_hf)
    lf = lf.view(batchsize_lf, -1, height_lf, width_lf)

    x = hf, lf
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add', alpha=0.5):
        
        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat
            
            # ensure output of concat has the same channels as 
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True
            )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = _BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from 
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False
            )


    @staticmethod
    def _add(x, out):
        # residual connection
        return x[0] + out[0], x[1] + out[1]

    @staticmethod
    def _concat(x, out):
        _cat = _Cat()
        # concatenate along channel axis
        return _cat((x,out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = _BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = _ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv


    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            residual = _avg_pool2d(residual, kernel_size=3, 
                stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        
        out = self._combine_func(residual, out)
        out = F.relu(out[0]), F.relu(out[1])
        return out


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        """ShuffleNet constructor.
        Arguments:
            groups (int, optional): number of groups to be used in grouped 
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.
        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels =  in_channels
        self.num_classes = num_classes

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(num_groups))
        
        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1], # stage 1
                             stride=2)
        self.maxpool = _MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)
        
        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2
        
        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
            )
        modules[stage_name+"_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'
                )
            modules[name] = module

        return nn.Sequential(modules)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # global average pooling layer
        x = _avg_pool2d(x, x.data.size()[-2:])

        return x


if __name__ == "__main__":
    """Testing
    """
model = ShuffleNet()