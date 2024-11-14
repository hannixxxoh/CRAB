import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from blocks import *

class ResidualBlockWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, reduction_ratio=16, pool_types=['avg', 'max'], kernel_size=3):
        super(ResidualBlockWithCBAM, self).__init__()

        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        self.cbam = CBAM(gate_channels=out_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

    def forward(self, x):
        x = self.residual_block(x)
        x = self.cbam(x)
        return x

class ResidualBlockWithSFB(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size=3):
        super(ResidualBlockWithSFB, self).__init__()

        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        self.sfb = SFB(embed_dim = 16)
    
    def forward(self, x):
        x = self.residual_block(x)
        x = self.sfb(x)
        return x

class ResidualBlockWithSFB_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size=3):
        super(ResidualBlockWithSFB_CBAM, self).__init__()

        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        self.sfb = SFB(embed_dim = 16, red=4)
        self.cbam = CBAM(gate_channels=out_channels, reduction_ratio=16, pool_types=['avg', 'max'])

    def forward(self, x):
        x = self.residual_block(x)
        x = self.sfb(x)
        x = self.cbam(x)
        return x

class ResidualBlockWithCoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size=3):
        super(ResidualBlockWithCoordAtt, self).__init__()

        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        self.coordatt = CoordAtt(inp=in_channels, oup=out_channels)

    def forward(self, x):
        x = self.residual_block(x)
        x = self.coordatt(x)
        return x        
    
class ResidualBlockWithGlobalContextWithRangeFusionAndSCA(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size=3):
        super(ResidualBlockWithGlobalContextWithRangeFusionAndSCA, self).__init__()

        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        self.GlobalContextWithRangeFusionAndSCA = GlobalContextWithRangeFusionAndSCA(c=16, GCE_Conv=3)

    def forward(self, input):
        x = self.residual_block(input)
        x = self.GlobalContextWithRangeFusionAndSCA(x)
        x = x + input
        return x        
    
class ResidualBlockWithFRB(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size=3):
        super(ResidualBlockWithFRB, self).__init__()

        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        self.frb = FreBlock9(channels=16)

    def forward(self, input):
        x = self.residual_block(input)
        x = self.frb(x)
        x = x + input
        return x        

class SESRRelease(nn.Module):
    """
    Collapsible Linear Blocks for Super-Efficient Super Resolution, Bhardwaj et al. (https://arxiv.org/abs/2103.09404)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_channels=16,
                 num_lblocks=3,
                 scaling_factor=2,
                 use_clipped_relu=False,
                 use_concat_residual=False):
        super().__init__()
        self.use_clipped_relu = use_clipped_relu
        self.use_concat_residual = use_concat_residual
        self.anchor = AnchorOp(scaling_factor)  # (=channel-wise nearest upsampling)

        self.conv_first = CollapsibleLinearBlock(in_channels=in_channels, out_channels=num_channels,
                                                        tmp_channels=256, kernel_size=5, activation='relu6')

        residual_layers = [
            ResidualCollapsibleLinearBlock(in_channels=num_channels, out_channels=num_channels,
                                                  tmp_channels=256, kernel_size=3, activation='relu6')
            for _ in range(num_lblocks)
        ]
        # self.residual_block = nn.ModuleList([
        #     #ResidualBlockWithCBAM(in_channels=num_channels, out_channels=num_channels, tmp_channels=256)
        #     ResidualBlockWithSFB_CBAM(in_channels=num_channels, out_channels=num_channels, tmp_channels=256)
        #     #ResidualBlockWithGlobalContextWithRangeFusionAndSCA(in_channels=num_channels, out_channels=num_channels, tmp_channels=256)
        #     #ResidualBlockWithFRB(in_channels=num_channels, out_channels=num_channels, tmp_channels=256)
        #     for _ in range(num_lblocks)
        # ])
        self.residual_block = nn.Sequential(*residual_layers)

        self.add_residual = AddOp()

        self.concat_residual = ConcatOp()

        if self.use_concat_residual:
            self.conv_last = CollapsibleLinearBlock(in_channels=num_channels*2,
                                                       out_channels=out_channels * scaling_factor ** 2,
                                                       tmp_channels=256, kernel_size=5, activation='identity')
        else:
            self.conv_last = CollapsibleLinearBlock(in_channels=num_channels,
                                                       out_channels=out_channels * scaling_factor ** 2,
                                                       tmp_channels=256, kernel_size=5, activation='identity')

        self.add_upsampled_input = AddOp()
        self.concat_upsampled_input = ConcatOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def collapse(self):
        self.conv_first.collapse()
        for layer in self.residual_block:
            layer.collapse()
        self.conv_last.collapse()

    def before_quantization(self):
        self.collapse()

    def forward(self, input):
        upsampled_input = self.anchor(input)  # Get upsampled input from AnchorOp()
        initial_features = self.conv_first(input)  # Extract features from conv-first
        #residual_features = self.residual_block(initial_features)  # Get residual features with `lblocks`

        residual_features = initial_features
        for layer in self.residual_block:
            residual_features = layer(residual_features)

        if self.use_concat_residual:
            residual_features = self.concat_residual(residual_features, initial_features)  # Add init_features and residual
        else:
            residual_features = self.add_residual(residual_features, initial_features)  # Add init_features and residual
        final_features = self.conv_last(residual_features)  # Get final features from conv-last
        output = self.add_upsampled_input(final_features, upsampled_input)  # Add final_features and upsampled_input

        if self.use_clipped_relu:
            output = self.depth_to_space(output) # Depth-to-space and return
            output = torch.clamp(output, min=0, max=1)
        else:
            output = self.depth_to_space(output)

        return output  # Depth-to-space and return


class SESRRelease_M3(SESRRelease):
    def __init__(self, scaling_factor, clipped_relu:bool=False, concat_residual:bool=False, **kwargs):
        super().__init__(scaling_factor=scaling_factor, num_channels=16, num_lblocks=3, use_clipped_relu=clipped_relu, use_concat_residual=concat_residual, **kwargs)


class SESRRelease_M5(SESRRelease):
    def __init__(self, scaling_factor, clipped_relu:bool=False, concat_residual:bool=False,**kwargs):
        super().__init__(scaling_factor=scaling_factor, num_channels=16, num_lblocks=5, use_clipped_relu=clipped_relu, use_concat_residual=concat_residual, **kwargs)


class SESRRelease_M7(SESRRelease):
    def __init__(self, scaling_factor, clipped_relu:bool=False, concat_residual:bool=False,**kwargs):
        super().__init__(scaling_factor=scaling_factor, num_channels=16, num_lblocks=7, use_clipped_relu=clipped_relu, use_concat_residual=concat_residual, **kwargs)


class SESRRelease_M11(SESRRelease):
    def __init__(self, scaling_factor, clipped_relu:bool=False, concat_residual:bool=False,**kwargs):
        super().__init__(scaling_factor=scaling_factor, num_channels=16, num_lblocks=11, use_clipped_relu=clipped_relu, use_concat_residual=concat_residual, **kwargs)

class SESRRelease_XL(SESRRelease):
    def __init__(self, scaling_factor, clipped_relu:bool=False, concat_residual:bool=False,**kwargs):
        super().__init__(scaling_factor=scaling_factor, num_channels=32, num_lblocks=11, use_clipped_relu=clipped_relu, concat_residual=concat_residual, **kwargs)

from clip_util import (ModifiedResNet,
                            UNetUpBlock,
                            UNetUpBlock_nocat,
                            conv3x3)

class ModifiedResNet_RemoveFinalLayer(ModifiedResNet):
   
    def __init__(self, layers, in_chn=3, width=64):
        super().__init__(layers, in_chn, width)

    def forward(self, x):
        out = []

        x = x.type(self.conv1.weight.dtype); out.append(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); out.append(x)
        x = self.avgpool(x)
        
        x = self.layer1(x); out.append(x)
        x = self.layer2(x); out.append(x)
        x = self.layer3(x); out.append(x)
        x = self.layer4(x); 

        return out

class CLIPSR(nn.Module):
    def __init__(self, num_blocks, inp_channels=3, scale=2, out_channels=3, depth=5, wf=64, slope=0.2,
                       bias=True, model_path=None, aug_level=0.05):

        super(CLIPSR, self).__init__()
        
        self.sigmas = [aug_level * i for i in range(1, 5)]
        
        self.inp_channels = inp_channels
        if inp_channels == 1: # used for 1 channel input, eg. CT images
            self.first = nn.Conv2d(inp_channels, 3, kernel_size=1, bias=bias)
            inp_channels = 3

        self.encoder = ModifiedResNet_RemoveFinalLayer(num_blocks, inp_channels, width=wf)
        self.encoder.load_pretrain_model_SR(model_path)
            
        for params in self.encoder.parameters():
            params.requires_grad = False

        # learnable decoder
        self.up_path = nn.ModuleList()
        prev_channels = wf * 2 ** (len(num_blocks))
        for i in range(depth):
            if i == 0:
                self.up_path.append(UNetUpBlock_nocat(prev_channels, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 2:
                self.up_path.append(UNetUpBlock(prev_channels*3//2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 1: # introduce noisy image as a dense feature
                self.up_path.append(UNetUpBlock(prev_channels+inp_channels, prev_channels, slope, bias))
            else:
                self.up_path.append(UNetUpBlock(prev_channels*2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2

        #self.last = conv3x3(prev_channels, out_channels * (2 ** 2), bias=bias)
        #self.depth_to_space = nn.PixelShuffle(2)
        self.upsampler = Upsampler(conv3x3, scale, prev_channels, bias=bias)
        self.last = conv3x3(prev_channels, out_channels, bias=bias)


    def forward(self, x):
        
        if self.inp_channels == 1:
            x = self.first(x)
            
        out = self.encoder(x)

        # progressive feature augmentation 
        if self.training:
            for idx in range(len(out)):
                if idx == 0: continue
                # alpha = torch.randn_like(out[idx]) * self.sigmas[idx-1] + 1.0
                # out[idx] = out[idx] * alpha
                
        x = out[-1]
         
        for i, up in enumerate(self.up_path):
            if i != 0: 
                x = up(x, out[-i-1])
            else:
                x = up(x)
        
        
        # print(x.shape)
        #x = self.depth_to_space(x)
        x = self.upsampler(x)
        #x = self.last(x)

        return x #self.last(x)


if __name__ == '__main__':
    #model = SESRRelease_M7(scaling_factor=2, clipped_relu=False, concat_residual=False)
    model = CLIPSR(num_blocks=[2, 2, 2, 2], inp_channels=3, out_channels=3, depth=5, wf=64, slope=0.2, bias=True, model_path=None, aug_level=0.05)
    model.eval()
    input_tensor = torch.randn(1, 3, 64, 64)
    output = model(input_tensor)
    print(output.shape)
