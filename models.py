'''
Created on Sep 3, 2017

@author: Michal.Busta at gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from octconv import OctConv2d
from torch.nn import LeakyReLU, Conv2d, Dropout2d, LogSoftmax, InstanceNorm2d

import math

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

class _Dropout2d(nn.Module):
  def __init__(self, p=0.5, inplace=False):
    super(_Dropout2d, self).__init__()
    self.dropout = Dropout2d(p=p, inplace=inplace)

  def forward(self, x):
    hf, hl = x
    return self.dropout(hf), self.dropout(hl)

class _InstanceNorm2d(nn.Module):
  def __init__(self, num_features, alpha_in=0.5, alpha_out=0.5, eps=1e-5, momentum=0.1, affine=True):
    super(_InstanceNorm2d, self).__init__()
    hf_ch = int(num_features * (1 - alpha_out))
    lf_ch = num_features - hf_ch
    self.inh = InstanceNorm2d(hf_ch, eps=eps, momentum=momentum, affine=affine)
    self.inl = InstanceNorm2d(lf_ch, eps=eps, momentum=momentum, affine=affine)

  def forward(self, x):
    hf, lf = x
    return self.inh(hf), self.inl(lf)

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

class _ReLU6(nn.Module):
  def __init__(self, inplace=False):
      super(_ReLU6, self).__init__()
      self.relu6 = nn.ReLU6(inplace=inplace)
  def forward(self, x):
    hf, lf = x
    return self.relu6(hf), self.relu6(lf)

class _LeakyReLU(nn.Module):
  def __init__(self, negative_slope=0.01, inplace=False):
    super(_LeakyReLU, self).__init__()
    self.negative_slope = negative_slope
    self.inplace = inplace
  def forward(self, x):
    hf, lf = x
    return F.leaky_relu(hf, negative_slope=self.negative_slope, inplace=self.inplace), F.leaky_relu(lf, negative_slope=self.negative_slope, inplace=self.inplace)

class _Interpolate(nn.Module):
  def forward(self, x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    hf, hl = x
    return F.interpolate(hf, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners), F.interpolate(hl, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),

class _Cat(nn.Module):
  def forward(self, x, dim=0):
    hf1, lf1 = x[0]
    hf2, lf2 = x[1]
    return torch.cat((hf1, hf2), dim), torch.cat((lf1, lf2), dim)

class _ReLU(nn.Module):
  def __init__(self, inplace=False):
    super(_ReLU, self).__init__()
    self.relu = nn.ReLU(inplace=inplace)
  def forward(self, x):
    hf, lf = x
    return self.relu(hf), self.relu(lf)

class CReLU(nn.Module):
  def __init__(self):
    super(CReLU, self).__init__()
    self.leaky_relu = _LeakyReLU(negative_slope=0.01, inplace=True)
    self.cat = _Cat()
  def forward(self, x):
    negative_x = -x[0], -x[1]
    return self.cat((self.leaky_relu(x), self.leaky_relu(negative_x)), 1)

class CReLU_IN(nn.Module):
  def __init__(self, channels):
    super(CReLU_IN, self).__init__()
    self.cat = _Cat()
    self.bn = _InstanceNorm2d(channels * 2, eps=1e-05, momentum=0.1, affine=True)
    self.leaky_relu = _LeakyReLU(negative_slope=0.01, inplace=True)

  def forward(self, x):
    negative_x = -x[0], -x[1]
    cat = self.cat((x, negative_x), 1)
    x = self.bn(cat)
    return self.leaky_relu(x)

def conv_bn(inp, oup, stride, alpha=0.5):
    return nn.Sequential(
      OctConv2d(inp, oup, 3, stride, 1, bias=False, alpha=alpha),
      _BatchNorm2d(oup),
      _ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride, dilation=1, alpha=0.5):
  if isinstance(alpha, int) or isinstance(alpha, float):
    alpha1 = alpha2 = alpha
  elif alpha is not None:
    alpha1 = alpha[0]
    alpha2 = alpha[1]
  return nn.Sequential(
    OctConv2d(inp, inp, 3, stride, 1 + (dilation > 0) * (dilation -1), dilation=dilation, groups=inp, bias=False, alpha=alpha1),
    _BatchNorm2d(inp),
    _LeakyReLU(inplace=True, negative_slope=0.01),

    OctConv2d(inp, oup, 1, 1, 0, bias=False, alpha=alpha2),
    _BatchNorm2d(oup),
    _LeakyReLU(inplace=True, negative_slope=0.01),
  )
  
def conv_dw_plain(inp, oup, stride, dilation=1, alpha=0.5):
  if isinstance(alpha, int) or isinstance(alpha, float):
    alpha1 = alpha2 = alpha
  elif alpha is not None:
    alpha1 = alpha[0]
    alpha2 = alpha[1]
  return nn.Sequential(
    OctConv2d(inp, inp, 3, stride, 1 + (dilation > 0) * (dilation -1), dilation=dilation, groups=inp, bias=False, alpha=alpha),
    OctConv2d(inp, oup, 1, 1, 0, bias=False, alpha=alpha)
  )
  
def conv_dw_res(inp, oup, stride, alpha=0.5):
  if isinstance(alpha, int) or isinstance(alpha, float):
    alpha1 = alpha2 = alpha
  elif alpha is not None:
    alpha1 = alpha[0]
    alpha2 = alpha[1]
  return nn.Sequential(
    OctConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, alpha=alpha1),
    _BatchNorm2d(inp),
    _LeakyReLU(inplace=True, negative_slope=0.01),

    OctConv2d(inp, oup, 1, 1, 0, bias=False, alpha=alpha2),
    _BatchNorm2d(oup),
  )

def conv_dw_in(inp, oup, stride, dilation=1, alpha = 0.5):
  if isinstance(alpha, int) or isinstance(alpha, float):
    alpha1 = alpha2 = alpha
  elif alpha is not None:
    alpha1 = alpha[0]
    alpha2 = alpha[1]
  return nn.Sequential(
    OctConv2d(inp, inp, 3, stride, 1 + (dilation > 0) * (dilation -1), dilation=dilation, groups=inp, bias=False, alpha=alpha1),
    OctConv2d(inp, oup, 1, 1, 0, bias=False, alpha=alpha2),
    _InstanceNorm2d(oup, eps=1e-05, momentum=0.1),
    _LeakyReLU(inplace=True, negative_slope=0.01),
  )

def conv_dw_res_in(inp, oup, stride, alpha=0.5):
  if isinstance(alpha, int) or isinstance(alpha, float):
    alpha1 = alpha2 = alpha
  elif alpha is not None:
    alpha1 = alpha[0]
    alpha2 = alpha[1]
  return nn.Sequential(
    OctConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, alpha=alpha1),
    _InstanceNorm2d(inp, eps=1e-05, momentum=0.1, affine=True),
    _LeakyReLU(inplace=True, negative_slope=0.01),

    OctConv2d(inp, oup, 1, 1, 0, bias=False, alpha=alpha2),
    _InstanceNorm2d(oup, eps=1e-05, momentum=0.1, affine=True)
  )
    
def dice_loss(inp, target):
    
  smooth = 1.
  iflat = inp.view(-1)
  tflat = target.view(-1)
  intersection = (iflat * tflat).sum()
  
  return - ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))

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

class BasicBlockSep(nn.Module):
  expansion = 1
  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, alpha=0.5):
    super(BasicBlockSep, self).__init__()
    self.conv_sep1 = conv_dw(inplanes, planes, stride, dilation=dilation, alpha=alpha)
    self.conv2 = conv_dw_res(planes, planes, 1, alpha=alpha)
    self.downsample = downsample
    self.stride = stride
    self.relu = _LeakyReLU(negative_slope=0.01, inplace=True)

  def forward(self, x):
    residual = x
    out = self.conv_sep1(x)
    out = self.conv2(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out = out[0]+residual[0], out[1]+residual[1]
    out = self.relu(out)
    return out 
      
class BasicBlockIn(nn.Module):
  expansion = 1
  def __init__(self, inplanes, planes, stride=1, downsample=None, alpha=0.5):
    super(BasicBlockIn, self).__init__()
    self.conv1 = OctConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, alpha=alpha)
    self.bn1 = _InstanceNorm2d(planes, eps=1e-05, momentum=0.1, affine=True)
    self.relu = _ReLU(inplace=True)
    self.conv2 = OctConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, alpha=alpha)
    self.bn2 = _InstanceNorm2d(planes, eps=1e-05, momentum=0.1, affine=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out = out[0]+residual[0], out[1]+residual[1]
    out = self.relu(out)

    return out
  
class BasicBlockSepIn(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, alpha=0.5):
    super(BasicBlockSepIn, self).__init__()
    
    self.conv_sep1 = conv_dw_in(inplanes, planes, stride, dilation=dilation, alpha=alpha)
    self.conv2 = conv_dw_res_in(planes, planes, 1, alpha=alpha)
    self.downsample = downsample
    self.stride = stride
    self.relu = _LeakyReLU(negative_slope=0.01, inplace=True)

  def forward(self, x):
    residual = x
    out = self.conv_sep1(x)
    out = self.conv2(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out = out[0]+residual[0], out[1]+residual[1]
    out = self.relu(out)
    return out
  
def iou_loss(roi_gt, byte_mask, roi_pred, box_loss_value):
  d1_gt = roi_gt[:, :, :, 0][byte_mask]
  d2_gt = roi_gt[:, :, :, 1][byte_mask] 
  d3_gt = roi_gt[:, :, :, 2][byte_mask]
  d4_gt = roi_gt[:, :, :, 3][byte_mask] 
  
  mask3 = torch.gt(d3_gt, 0)   
  mask4 = torch.gt(d4_gt, 0)   
  d3_gt = d3_gt[mask3]
  d4_gt = d4_gt[mask4] 
  
  
  d1_pred = roi_pred[:, 0, :, :][byte_mask]
  d2_pred = roi_pred[:, 1, :, :][byte_mask]
  d3_pred = roi_pred[:, 2, :, :][byte_mask]
  d3_pred = d3_pred[mask3]
  d4_pred = roi_pred[:, 3, :, :][byte_mask]
  d4_pred = d4_pred[mask4]
  
  area_gt_l = (d1_gt[mask3] + d2_gt[mask3]) * (d3_gt)
  area_pred_l = (d1_pred[mask3] + d2_pred[mask3]) * (d3_pred)
  w_union_l = torch.min(d3_gt, d3_pred)
  h_union_l = torch.min(d1_gt[mask3], d1_pred[mask3]) + torch.min(d2_gt[mask3], d2_pred[mask3])
  area_intersect_l = w_union_l * h_union_l
  area_union_l = area_gt_l + area_pred_l - area_intersect_l
  AABB_l = - torch.log((area_intersect_l + 1.0)/(area_union_l + 1.0))
  
  if AABB_l.dim() > 0:
    box_loss_value += torch.mean(AABB_l)
  
  area_gt_r = (d1_gt[mask4] + d2_gt[mask4]) * (d4_gt)
  area_pred_r = (d1_pred[mask4] + d2_pred[mask4]) * (d4_pred)
  w_union_r = torch.min(d4_gt, d4_pred)
  h_union_r = torch.min(d1_gt[mask4], d1_pred[mask4]) + torch.min(d2_gt[mask4], d2_pred[mask4])
  area_intersect_r = w_union_r * h_union_r
  area_union_r = area_gt_r + area_pred_r - area_intersect_r
  AABB_r = - torch.log((area_intersect_r + 1.0)/(area_union_r + 1.0))
  if AABB_r.dim() > 0:
    box_loss_value += torch.mean(AABB_r)
  
class OctShuffleMLT(nn.Module):
  
  def recompute(self):
    self.layer0[0].recompute_weights()
    self.layer0[2].recompute_weights()
    self.layer0_1[0].recompute_weights()
    self.layer0_1[2].recompute_weights()
            
  def __init__(self, attention = False, multi_scale = True):
    super(OctShuffleMLT, self).__init__()
    
    self.inplanes = 64
    alpha = 0.5
    
    self.layer0 = nn.Sequential(
      OctConv2d(3, 16, 3, stride=1, padding=1, bias=False, alpha=(0, 0.5)),
      CReLU_IN(16),
      OctConv2d(32, 32, 3, stride=2, padding=1, bias=False),
      CReLU_IN(32)
    )
    
    self.layer0_1 = nn.Sequential(
      OctConv2d(64, 64, 3, stride=1, padding=1, bias=False),
      #nn.InstanceNorm2d(64, affine=True),
      _ReLU(),
      OctConv2d(64, 64, 3, stride=2, padding=1, bias=False),
      #nn.InstanceNorm2d(64, affine=True),
      _ReLU(inplace=True)
    )
    
    self.conv5 = OctConv2d(64, 128, 3, padding=1, bias=False)
    self.conv6 = OctConv2d(128, 128, 3, padding=1, bias=False)
    self.conv7 = OctConv2d(128,256, 3, padding=1, bias=False)
    self.conv8 = OctConv2d(256, 256, 3, padding=1, bias=False)
    self.conv9_1 = OctConv2d(256, 256, 3, padding=1, bias=False)
    self.conv9_2 = OctConv2d(256, 256, 3, padding=1, bias=False, alpha=(0.5, 0))
    self.conv10_s = Conv2d(256, 256, (2, 3), padding=(0, 1), bias=False)
    self.conv11 = Conv2d(256, 8400, 1, padding=(0,0))
    
    self.batch5 = _InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    self.batch6 = _InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    self.batch7 = _InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    self.batch8 = _InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    self.batch9 = _InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    self.batch10_s = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    self.max2_1 = nn.MaxPool2d((2, 1), stride=(2,1))
    self.max2 = _MaxPool2d((2, 1), stride=(2,1))
    self.leaky = _LeakyReLU(negative_slope=0.01, inplace=True)
    self.leaky2 = LeakyReLU(negative_slope=0.01, inplace=True)

    self.groups = 3
    self.stage_out_channels = [-1, 24, 240, 480, 960]
    self.stage_repeats = [3, 7, 3]
    self.layer1 = self._make_layer(BasicBlockIn, 24, 3, stride=1, alpha=alpha)
    self.layer2 = self._make_stage(2)
    self.layer3 = self._make_stage(3)
    self.layer4 = self._make_stage(4)

    self.feature4 = OctConv2d(960, 256, 1, stride=1, padding=0, bias=False, alpha=(0.5, 0))
    self.feature3 = OctConv2d(480, 256, 1, stride=1, padding=0, bias=False, alpha=(0.5, 0))
    self.feature2 = OctConv2d(240, 256, 1, stride=1, padding=0, bias=False, alpha=(0.5, 0))
    
    self.upconv2 = conv_dw_plain(256, 256, stride=1, alpha=0)
    self.upconv1 = conv_dw_plain(256, 256, stride=1, alpha=0)
    
    self.feature1 = OctConv2d(24, 256, 1, stride=1, padding=0, bias=False, alpha=(0.5, 0))
    
    self.act = OctConv2d(256, 1, 1, padding=0, stride=1, alpha=0)
    self.rbox = OctConv2d(256, 4, 1, padding=0, stride=1, alpha=0)
    
    self.angle = OctConv2d(256, 2, 1, padding=0, stride=1, alpha=0)
    self.drop0 = _Dropout2d(p=0.2, inplace=False)
    self.drop1 = Dropout2d(p=0.2, inplace=False)
    
    self.angle_loss = nn.MSELoss(reduction='elementwise_mean')
    self.h_loss = nn.SmoothL1Loss(reduction='elementwise_mean')
    self.w_loss = nn.SmoothL1Loss(reduction='elementwise_mean')
    
    self.attention = attention
  
    if self.attention:
      self.conv_attenton = OctConv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True, alpha=0) 
    
    self.multi_scale = multi_scale

  def _make_layer(self, block, planes, blocks, stride=1, alpha=0.5):
    
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
  
        OctConv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False, alpha=alpha),
        _BatchNorm2d(planes * block.expansion, alpha_in=alpha, alpha_out=alpha),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, alpha=alpha))

    return nn.Sequential(*layers)
  
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
  
  def forward_ocr(self, x):
    
    x = self.conv5(x)
    x = self.batch5(x)
    x = self.leaky(x)
    
    x = self.conv6(x)
    x = self.leaky(x)
    x = self.conv6(x)
    x = self.leaky(x)
    
    x = self.max2(x)
    x = self.conv7(x)
    x = self.batch7(x)
    x = self.leaky(x)
    
    
    x = self.conv8(x)
    x = self.leaky(x)
    x = self.conv8(x)
    x = self.leaky(x)
    
    x = self.conv9_1(x)
    x = self.leaky(x)
    x = self.conv9_2(x)
    x = self.leaky2(x)
    
    x = self.max2_1(x)
    
    x = self.conv10_s(x)
    x = self.batch10_s(x)
    x = self.leaky2(x)
    
    
    x = self.drop1(x)
    x = self.conv11(x)
    x = x.squeeze(2)

    x = x.permute(0,2,1)
    y = x
    x = x.contiguous().view(-1,x.data.shape[2])
    x = LogSoftmax(len(x.size()) - 1)(x)
    x = x.view_as(y)
    x = x.permute(0,2,1)
    
    return x   
  
  def forward_features(self, x):
    
    x = self.layer0(x)
    focr = self.layer0_1(x)
    return focr

  def forward(self, x):
    
    x = self.layer0(x)
    x = self.layer0_1(x)
    
    x = self.drop0(x)
    su3 = self.layer1(x)
    features1 = self.feature1(su3)
    su2 = self.layer2(su3)
    features2 = self.feature2(su2)
    su1 = self.layer3(su2)
    features3 = self.feature3(su1)
    x = self.layer4(su1)
    
    x = self.drop0(x)
    
    features4 = self.feature4(x)
    if self.attention:
      att = self.conv_attenton(features4)
      att = torch.sigmoid(att)
      att = att.expand_as(features4)
      att_up = F.interpolate(att, size=(features3.size(2), features3.size(3)), mode='bilinear', align_corners=True)
    
    x = F.interpolate(features4, size=(features3.size(2), features3.size(3)), mode='bilinear', align_corners=True)
    
    if self.attention:
      x = x + features3 * att_up
      att = self.conv_attenton(x)
      att = torch.sigmoid(att)
      att_up = F.interpolate(att, size=(features2.size(2), features2.size(3)), mode='bilinear', align_corners=True)
    else:
      x = x + features3 
      
    x = F.interpolate(x, size=(features2.size(2), features2.size(3)), mode='bilinear', align_corners=True)
    x = self.upconv1(x)
    if self.attention:
      features2 = x + features2 * att_up
      att = self.conv_attenton(features2)
      att = torch.sigmoid(att)
      att_up = F.interpolate(att, size=(features1.size(2), features1.size(3)), mode='bilinear', align_corners=True)
    else:
      features2 = x + features2  
    x = features2
        
    x = F.interpolate(x, size=(features1.size(2), features1.size(3)), mode='bilinear', align_corners=True)
    x = self.upconv2(x)
    
    if self.attention:
      x = x + features1 * att_up
    else:
      x += features1
    
    segm_pred2 = torch.sigmoid(self.act(features2))
    rbox2 = torch.sigmoid(self.rbox(features2)) * 128
    angle2 = torch.sigmoid(self.angle(features2)) * 2 - 1 
    angle_den = torch.sqrt(angle2[:, 0, :, :] * angle2[:, 0, :, :] + angle2[:, 1, :, :] * angle2[:, 1, :, :]).unsqueeze(1)
    angle_den = angle_den.expand_as(angle2)
    angle2 = angle2 / angle_den
    
    x = self.drop1(x)
    
    segm_pred = torch.sigmoid(self.act(x))
    rbox = torch.sigmoid(self.rbox(x)) * 128
    angle = torch.sigmoid(self.angle(x)) * 2 - 1 
    angle_den = torch.sqrt(angle[:, 0, :, :] * angle[:, 0, :, :] + angle[:, 1, :, :] * angle[:, 1, :, :]).unsqueeze(1)
    angle_den = angle_den.expand_as(angle)
    angle = angle / angle_den
    
    return [segm_pred, segm_pred2], [rbox, rbox2], [angle, angle2], x

  def loss(self, segm_preds, segm_gt, iou_mask, angle_preds, angle_gt, roi_pred, roi_gt):
    
    self.box_loss_value =  torch.tensor(0.0, requires_grad = True).cuda()
    self.angle_loss_value =  torch.tensor(0.0, requires_grad = True).cuda()
  
    segm_pred = segm_preds[0].squeeze(1)
    angle_pred = angle_preds[0]
    self.segm_loss_value = dice_loss(segm_pred * iou_mask , segm_gt * iou_mask )
    segm_pred1 = segm_preds[1].squeeze(1)
    
    if self.multi_scale:
      iou_gts = F.interpolate(segm_gt.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True).squeeze(1)
      iou_masks = F.interpolate(iou_mask.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True).squeeze(1)
      self.segm_loss_value += dice_loss(segm_pred1 * iou_masks, iou_gts * iou_masks )
      
    byte_mask = torch.gt(segm_gt, 0.5)
    
    if byte_mask.sum() > 0:
      
      gt_sin = torch.sin(angle_gt[byte_mask])
      gt_cos = torch.cos(angle_gt[byte_mask])
      
      sin_val = self.angle_loss(angle_pred[:, 0, :, :][byte_mask], gt_sin)
      cos_val = self.angle_loss(angle_pred[:, 1, :, :][byte_mask], gt_cos)
       
      self.angle_loss_value += sin_val
      self.angle_loss_value += cos_val
      
      iou_loss(roi_gt, byte_mask, roi_pred[0], self.box_loss_value)
        
      if self.multi_scale:
        byte_mask = torch.gt(F.interpolate(segm_gt.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True), 0.5).squeeze(1)
        if byte_mask.sum() > 0:
          
          angle_gts = F.interpolate(angle_gt.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True).squeeze(1)
          gt_sin = torch.sin(angle_gts[byte_mask])
          gt_cos = torch.cos(angle_gts[byte_mask])
          sin_val = self.angle_loss(angle_preds[1][:, 0, :, :][byte_mask], gt_sin)
          
          self.angle_loss_value += sin_val
          self.angle_loss_value += self.angle_loss(angle_preds[1][:, 1, :, :][byte_mask], gt_cos)
          
          roi_gt_s = F.interpolate(roi_gt.permute(0, 3, 1, 2), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True) / 2
          roi_gt_s = roi_gt_s.permute(0, 2, 3, 1)
          iou_loss(roi_gt_s, byte_mask, roi_pred[1], self.box_loss_value)
            
    return self.segm_loss_value +  self.angle_loss_value * 2 + 0.5 * self.box_loss_value