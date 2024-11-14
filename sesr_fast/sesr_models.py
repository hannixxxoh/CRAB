import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from blocks import *

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
                                                        tmp_channels=256, kernel_size=5, activation='relu')

        residual_layers = [
            ResidualCollapsibleLinearBlock(in_channels=num_channels, out_channels=num_channels,
                                                  tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
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
        residual_features = self.residual_block(initial_features)  # Get residual features with `lblocks`
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