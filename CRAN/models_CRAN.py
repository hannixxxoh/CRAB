import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from blocks_CRAN import *

# Compressed Residual Attention Network ( CRAN)
class CRAN(nn.Module):
    """
    Collapsible Linear Blocks for Super-Efficient Super Resolution, Bhardwaj et al. (https://arxiv.org/abs/2103.09404)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_channels=16,
                 num_lblocks=3,
                 scaling_factor=2):
        super().__init__()
        self.anchor = AnchorOp(scaling_factor)  # (=channel-wise nearest upsampling)

        self.conv_first = CompressedLinearBlock(in_channels=in_channels, out_channels=num_channels,
                                                        tmp_channels=256, kernel_size=5, activation='relu')


        residual_layers = [
            CompressedResidualAttentionBlock(in_channels=num_channels, out_channels=num_channels,
                                                  tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        self.residual_block = nn.Sequential(*residual_layers)

        self.add_residual = AddOp()

        self.conv_last = CompressedLinearBlock(in_channels=num_channels,
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

        residual_features = self.add_residual(residual_features, initial_features)  # Add init_features and residual
        final_features = self.conv_last(residual_features)  # Get final features from conv-last
        output = self.add_upsampled_input(final_features, upsampled_input)  # Add final_features and upsampled_input
        output = self.depth_to_space(output) # Depth-to-space and return
        # output = self.add_upsampled_input(output, upsampled_input)

        output = torch.clamp(output, min=0, max=1)


        return output  # Depth-to-space and return

class CRAN_M7(CRAN):
    def __init__(self, scaling_factor):
        super().__init__(scaling_factor=scaling_factor, num_channels=16, num_lblocks=7)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_model(model):
    # 각 Conv2d 레이어의 파라미터 확인
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # Conv2d 레이어만 필터링
            layer_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += layer_params
            print(f"Layer: {name} | Params: {layer_params} | Shape: {list(module.parameters())[0].shape}")

    print(f"Total Conv2d Parameters: {total_params}")

from torchsummary import summary

if __name__ == '__main__':
    net = CRAN_M7(scaling_factor=2)
    net.collapse()
    print(count_model(net))

    # model.eval()
    # model.collapse()
    # print(model)
    # print(count_parameters(model))
    # 512x512 이미지를 랜덤하게 생성
    input_image = torch.rand(1, 3, 256, 256)  # 배치 크기 1, RGB 이미지 (채널, 높이, 너비)

    # 모델에 입력하여 결과를 얻음
    with torch.no_grad():  # 그래디언트 계산 비활성화
        output_image = net(input_image)

    # 결과 출력 (예: 출력 이미지의 크기)
    print("Output image shape:", output_image.shape)
