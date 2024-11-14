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
import math


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



class CollapsibleLinearBlock(nn.Module):
    """
    A convolutional block that can be collapsed into a single conv layer at inference.

     References:
         - Collapsible Linear Blocks for Super-Efficient Super Resolution, Bhardwaj et al.
           https://arxiv.org/abs/2103.09404
    """

    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, activation='prelu'):
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
        elif activation == 'relu6':
            self.activation = nn.ReLU6()
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


class ResidualCollapsibleLinearBlock(CollapsibleLinearBlock):
    """
    Residual version of CollapsibleLinearBlock.
    """

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(x + self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return
        super().collapse()
        middle = self.conv_expand.kernel_size[0] // 2
        num_channels = self.conv_expand.in_channels
        with torch.no_grad():
            for idx in range(num_channels):
                self.conv_expand.weight[idx, idx, middle, middle] += 1.

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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.last_conv = last_conv

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv2 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)

        if self.last_conv:
            self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        if self.last_conv:
            output = self.last_conv(output)
        return output


## Residual Block (RB)
class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


class SFB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(SFB, self).__init__()
        self.S = ResB(embed_dim, red)
        self.F = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def __call__(self, x):
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.fusion(out)
        return out
    

###########################################################################################
#CGNet
###########################################################################################

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=0, stride=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UpsampleWithFlops(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
        self.__flops__ = 0

    def forward(self, input):
        self.__flops__ += input.numel()
        return super(UpsampleWithFlops, self).forward(input)

class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False):
        super(GlobalContextExtractor, self).__init__()
        self.depthwise_separable_convs = nn.ModuleList([
            depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])

    def forward(self, x):
        outputs = []
        for conv in self.depthwise_separable_convs:
            x = F.gelu(conv(x))
            outputs.append(x)
        return outputs

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimplifiedChannelAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        attention = self.global_pool(x)
        attention = self.conv(attention)
        return x * attention

class GlobalContextWithRangeFusionAndSCA(nn.Module):
    def __init__(self, c, GCE_Conv=2):
        super().__init__()
        self.dw_channel = c
        self.GCE_Conv = GCE_Conv
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True)

        if self.GCE_Conv == 3:
            self.GCE = GlobalContextExtractor(c=self.dw_channel // 2, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])
        else:
            self.GCE = GlobalContextExtractor(c=self.dw_channel // 2, kernel_sizes=[3, 3], strides=[2, 3])

        #self.upsample = UpsampleWithFlops(scale_factor=2, mode='nearest')  # scale_factor로 변경
        
        # Simplified Channel Attention
        self.sca = SimplifiedChannelAttention(in_channels=int(self.dw_channel * (2 if GCE_Conv == 2 else 2.5)))

        # 1x1 Conv layer
        self.project_out = nn.Conv2d(int(self.dw_channel * (2 if GCE_Conv == 2 else 2.5)), c, kernel_size=1, padding=0, stride=1)


    def forward(self, x):
        b, c, h, w = x.shape
        self.upsample = nn.Upsample(size=(h, w), mode='nearest')
        x = self.conv1(x)
        x = self.conv2(x)
        x_1, x_2 = x.chunk(2, dim=1)
        
        # Global Context Extractor와 Range Fusion 적용
        if self.GCE_Conv == 3:
            x1, x2, x3 = self.GCE(x_1 + x_2)
            #print(self.upsample(x1).shape, self.upsample(x2).shape, self.upsample(x3).shape) 
            x = torch.cat([x, self.upsample(x1), self.upsample(x2), self.upsample(x3)], dim=1)
        else:
            x1, x2 = self.GCE(x_1 + x_2)
            x = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim=1)
        
        # Simplified Channel Attention 적용
        x = self.sca(x)
        
        # 1x1 Conv를 사용해 원래 채널 수로 프로젝션
        x = self.project_out(x)
        
        return x


class FreBlock9(nn.Module):
    def __init__(self, channels):
        super(FreBlock9, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        out = out + x
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



if __name__ == '__main__':
    model = FreBlock9(channels=16)
    input_data = torch.randn(1, 16, 128, 128)  # Input shape: (1, 64, 128, 128)
    output = model(input_data)
    print(output.shape)  # Output shape: (1, 64, 128, 128)