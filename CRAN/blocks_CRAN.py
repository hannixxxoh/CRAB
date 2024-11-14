# pylint: skip-file
#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        # find max attention and mean attention of all channels
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2


class ConcatOp(nn.Module):
    def forward(self, *args):
        return torch.cat([*args], dim=1)

class AnchorOp(nn.Module):
    """
    Repeat interleaves the input scaling_factor**2 number of times along the channel axis.
    """
    def __init__(self, scaling_factor, in_channels=3, init_weights=True, freeze_weights=True, kernel_size=1, **kwargs):
        """
        Args:
            scaling_factor: Scaling factor
            init_weights:   Initializes weights to perform nearest upsampling (Default for Anchor)
            freeze_weights:         Whether to freeze weights (if initialised as nearest upsampling weights)
        """
        super().__init__()

        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=(in_channels * scaling_factor**2),
                             kernel_size=kernel_size,
                             **kwargs)

        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, ii % num_channels_per_group,
                kernel_size // 2, kernel_size // 2] = 1.

            new_state_dict = OrderedDict({'weight': weight, 'bias': bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True) # [batch, channel, 1, 1]
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5) # [batch, channel, 1, 1]

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# contrast-aware spatial attention module
class CSALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool_h = nn.AdaptiveAvgPool2d(2)
        self.avg_pool_w = nn.AdaptiveAvgPool2d(3)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CompressedLinearBlock_stride(nn.Module):

    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, padding, stride=1, activation='relu'):
        super().__init__()

        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tmp_channels = tmp_channels

        self.conv_expand = nn.Conv2d(in_channels, tmp_channels, (kernel_size, kernel_size), stride=self.stride,
                                     padding=self.padding, bias=False)
        self.conv_squeeze = nn.Conv2d(tmp_channels, out_channels, (1, 1))

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise Exception(f'Activation not supported: {activation}')

        self.collapsed=False

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return

        new_conv = nn.Conv2d(self.conv_expand.in_channels,
                             self.conv_squeeze.out_channels,
                             self.conv_expand.kernel_size,
                             stride=self.stride,
                             padding=self.padding)

        # Find corresponding kernel weights by applying the convolutional block
        # to a delta function (center=1, 0 everywhere else)
        delta = torch.eye(self.conv_expand.in_channels)
        delta = delta.unsqueeze(2).unsqueeze(3)
        k = self.conv_expand.kernel_size[0]
        pad = int((k - 1) / 2)  # note: this will probably break if k is even
        delta = F.pad(delta, (pad, pad, pad, pad))  # Shape: in_channels x in_channels x kernel_size x kernel_size
        delta = delta.to(self.conv_expand.weight.device)

        padding = int((self.conv_expand.kernel_size[0] - 1)/ 2)

        temp_conv = nn.Conv2d(self.in_channels, self.tmp_channels, (self.conv_expand.kernel_size[0], self.conv_expand.kernel_size[1]),
                                        padding=padding, stride=1, bias=False)
        temp_conv.weight.data = self.conv_expand.weight.data.clone()


        with torch.no_grad():
            bias = self.conv_squeeze.bias
            kernel_biased = self.conv_squeeze(temp_conv(delta))
            kernel = kernel_biased - bias[None, :, None, None]

        # Flip and permute
        kernel = torch.flip(kernel, [2, 3])
        kernel = kernel.permute([1, 0, 2, 3])

        # Assign weight and return
        new_conv.weight = nn.Parameter(kernel)
        new_conv.bias = bias

        # Replace current layers
        self.conv_expand = new_conv
        self.conv_squeeze = nn.Identity()


        self.collapsed = True

class CompressedResidualBlock_stride(nn.Module):

    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, padding, stride=1, activation='relu'):
        super().__init__()

        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tmp_channels = tmp_channels

        self.conv_expand = nn.Conv2d(in_channels, tmp_channels, (kernel_size, kernel_size), stride=self.stride,
                                     padding=self.padding, bias=False)
        self.conv_squeeze = nn.Conv2d(tmp_channels, out_channels, (1, 1))

        self.initialize()

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise Exception(f'Activation not supported: {activation}')

        self.collapsed=False

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return

        new_conv = nn.Conv2d(self.conv_expand.in_channels,
                             self.conv_squeeze.out_channels,
                             self.conv_expand.kernel_size,
                             stride=self.stride,
                             padding=self.padding)

        # Find corresponding kernel weights by applying the convolutional block
        # to a delta function (center=1, 0 everywhere else)
        delta = torch.eye(self.conv_expand.in_channels)
        delta = delta.unsqueeze(2).unsqueeze(3)
        k = self.conv_expand.kernel_size[0]
        pad = int((k - 1) / 2)  # note: this will probably break if k is even
        delta = F.pad(delta, (pad, pad, pad, pad))  # Shape: in_channels x in_channels x kernel_size x kernel_size
        delta = delta.to(self.conv_expand.weight.device)

        padding = int((self.conv_expand.kernel_size[0] - 1)/ 2)

        temp_conv = nn.Conv2d(self.in_channels, self.tmp_channels, (self.conv_expand.kernel_size[0], self.conv_expand.kernel_size[1]),
                                        padding=padding, stride=1, bias=False)
        temp_conv.weight.data = self.conv_expand.weight.data.clone()


        with torch.no_grad():
            bias = self.conv_squeeze.bias
            kernel_biased = self.conv_squeeze(temp_conv(delta))
            kernel = kernel_biased - bias[None, :, None, None]

        # Flip and permute
        kernel = torch.flip(kernel, [2, 3])
        kernel = kernel.permute([1, 0, 2, 3])

        # Assign weight and return
        new_conv.weight = nn.Parameter(kernel)
        new_conv.bias = bias

        # Replace current layers
        self.conv_expand = new_conv
        self.conv_squeeze = nn.Identity()


        self.collapsed = True

    def initialize(self):
        middle = self.conv_expand.kernel_size[0] // 2
        num_residual_channels_expand = min(self.conv_expand.in_channels, self.conv_expand.out_channels)
        with torch.no_grad():
            for idx in range(num_residual_channels_expand):
                self.conv_expand.weight[idx, idx, middle, middle] += 1.

class CompressedLinearBlock(nn.Module):

    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, activation='relu'):
        super().__init__()

        self.conv_expand = nn.Conv2d(in_channels, tmp_channels, (kernel_size, kernel_size),
                                     padding=int((kernel_size - 1) / 2), bias=False)
        self.conv_squeeze = nn.Conv2d(tmp_channels, out_channels, (1, 1))

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'clipped':
            self.activation = nn.Hardtanh(0, 1)  # Clipped ReLU
        else:
            raise Exception(f'Activation not supported: {activation}')

        self.collapsed=False

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return

        padding = int((self.conv_expand.kernel_size[0] - 1)/ 2)
        new_conv = nn.Conv2d(self.conv_expand.in_channels,
                             self.conv_squeeze.out_channels,
                             self.conv_expand.kernel_size,
                             padding=padding)

        # Find corresponding kernel weights by applying the convolutional block
        # to a delta function (center=1, 0 everywhere else)
        delta = torch.eye(self.conv_expand.in_channels)
        delta = delta.unsqueeze(2).unsqueeze(3)
        k = self.conv_expand.kernel_size[0]
        pad = int((k - 1) / 2)  # note: this will probably break if k is even
        delta = F.pad(delta, (pad, pad, pad, pad))  # Shape: in_channels x in_channels x kernel_size x kernel_size
        delta = delta.to(self.conv_expand.weight.device)

        with torch.no_grad():
            bias = self.conv_squeeze.bias
            kernel_biased = self.conv_squeeze(self.conv_expand(delta))
            kernel = kernel_biased - bias[None, :, None, None]

        # Flip and permute
        kernel = torch.flip(kernel, [2, 3])
        kernel = kernel.permute([1, 0, 2, 3])

        # Assign weight and return
        new_conv.weight = nn.Parameter(kernel)
        new_conv.bias = bias

        # Replace current layers
        self.conv_expand = new_conv
        self.conv_squeeze = nn.Identity()


        self.collapsed = True

class CompressedResidualBlock(CompressedLinearBlock):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, activation='clipped'):
        super().__init__(in_channels, out_channels, tmp_channels, kernel_size, activation)
        self.initialize()

    def forward(self, x):
        if self.collapsed:
            out = self.conv_expand(x)
            return self.activation(out)
        out = self.conv_squeeze(self.conv_expand(x))
        return self.activation(out)

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return
        super().collapse()

    def initialize(self):
        middle = self.conv_expand.kernel_size[0] // 2
        num_residual_channels_expand = min(self.conv_expand.in_channels, self.conv_expand.out_channels)
        with torch.no_grad():
            for idx in range(num_residual_channels_expand):
                self.conv_expand.weight[idx, idx, middle, middle] += 1.

# class CompressedESA(nn.Module):
#     def __init__(self, n_feats, conv):
#         super(CompressedESA, self).__init__()
#         f = n_feats // 4
#         self.conv1 = conv(n_feats, f, kernel_size=1)
#         self.conv_max = conv(f, f, kernel_size=3, padding=1)
#         # self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
#         self.conv2 = CompressedResidualBlock_stride(f, f, f**3, kernel_size=3, stride=2, padding=0)
#         self.conv3 = conv(f, f, kernel_size=3, padding=1)
#         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
#         self.conv_f = conv(f, f, kernel_size=1)
#         self.conv4 = conv(f, n_feats, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         c1_ = (self.conv1(x)) # conv-1
#         c1 = self.conv2(c1_) # strided conv
#         v_max = F.max_pool2d(c1, kernel_size=7, stride=3) # max pooling
#         v_range = self.relu(self.conv_max(v_max)) # conv groups
#         c3 = self.relu(self.conv3(v_range)) # conv groups
#         c3 = self.conv3_(c3) # conv groups
        
#         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) # upsampling
#         cf = self.conv_f(c1_) # conv after upsampling
#         c4 = self.conv4(c3+cf) # conv-1
#         m = self.sigmoid(c4)
        
#         return x * m

class CompressedESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(CompressedESA, self).__init__()
        f = n_feats // 4
        # self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv1 = CompressedLinearBlock(n_feats, f, n_feats**2, kernel_size=3, activation='identity')
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        # self.conv_f = CompressedLinearBlock(f, f, f**4, kernel_size=3, activation='identity')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x)) # conv-1
        c1 = self.conv2(c1_) # strided conv

        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) # max pooling

        v_range = self.relu(self.conv_max(v_max)) # conv groups
        c3 = self.relu(self.conv3(v_range)) # conv groups
        c3 = self.conv3_(c3) # conv groups
        
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) # upsampling
        cf = self.conv_f(c1_) # conv after upsampling
        c4 = self.conv4(c3+cf) # conv-1
        m = self.sigmoid(c4)
        
        return x * m

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x)) # conv-1
        c1 = self.conv2(c1_) # strided conv

        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) # max pooling

        v_range = self.relu(self.conv_max(v_max)) # conv groups
        c3 = self.relu(self.conv3(v_range)) # conv groups
        c3 = self.conv3_(c3) # conv groups
        
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) # upsampling
        cf = self.conv_f(c1_) # conv after upsampling
        c4 = self.conv4(c3+cf) # conv-1
        m = self.sigmoid(c4)
        
        return x * m


class SpatialChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialChannelAttentionBlock, self).__init__()
        self.esa = ESA(in_channels, nn.Conv2d)
        self.cca = CCALayer(in_channels)
    def forward(self, out, x):
        out = self.esa(out + x)
        out = self.cca(out)

        return out

class CompressedResidualAttentionBlock(CompressedLinearBlock):

    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, activation='clipped'):
        super().__init__(in_channels, out_channels, tmp_channels, kernel_size, activation)
        self.initialize()
        self.spatial_channel_attention = SpatialChannelAttentionBlock(out_channels)

    def forward(self, x):
        if self.collapsed:
            out = self.conv_expand(x)
            out = self.spatial_channel_attention(out, x)
            return self.activation(out)
        out = self.conv_squeeze(self.conv_expand(x))
        out = self.spatial_channel_attention(out, x)
        return self.activation(out)

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return
        super().collapse()

    def initialize(self):
        middle = self.conv_expand.kernel_size[0] // 2
        num_residual_channels_expand = min(self.conv_expand.in_channels, self.conv_expand.out_channels)
        with torch.no_grad():
            for idx in range(num_residual_channels_expand):
                self.conv_expand.weight[idx, idx, middle, middle] += 1.
