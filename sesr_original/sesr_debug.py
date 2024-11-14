import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from blocks import *

class ResidualBlockWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, reduction_ratio=16, pool_types=['avg', 'max'], kernel_size=3):
        super(ResidualBlockWithCBAM, self).__init__()

        # ResidualCollapsibleLinearBlock 정의
        self.residual_block = ResidualCollapsibleLinearBlock(
            in_channels=in_channels, out_channels=out_channels, tmp_channels=tmp_channels, kernel_size=kernel_size, activation='relu'
        )

        # CBAM 정의
        self.cbam = CBAM(gate_channels=out_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

    def forward(self, x):
        # Residual 블록 통과
        x = self.residual_block(x)
        # CBAM 통과
        x = self.cbam(x)
        return x

class SESRRelease(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_channels=16,
                 num_lblocks=3,
                 scaling_factor=2,
                 use_clipped_relu=False,
                 use_concat_residual=False):
        super(SESRRelease, self).__init__()

        # 전달된 scaling_factor를 처리할 수 있도록 멤버 변수로 저장
        self.scaling_factor = scaling_factor
        self.use_clipped_relu = use_clipped_relu
        self.use_concat_residual = use_concat_residual
        self.anchor = AnchorOp(scaling_factor)  # (=channel-wise nearest upsampling)

        # conv-first 레이어 정의
        self.conv_first = CollapsibleLinearBlock(in_channels=in_channels, out_channels=num_channels,
                                                 tmp_channels=256, kernel_size=5, activation='relu')

        # Residual 블록에 CBAM 추가
        self.residual_layers = nn.ModuleList([
            ResidualBlockWithCBAM(in_channels=num_channels, out_channels=num_channels, tmp_channels=256)
            for _ in range(num_lblocks)
        ])

        self.add_residual = AddOp()
        self.concat_residual = ConcatOp()

        if self.use_concat_residual:
            self.conv_last = CollapsibleLinearBlock(in_channels=num_channels * 2,
                                                    out_channels=out_channels * scaling_factor ** 2,
                                                    tmp_channels=256, kernel_size=5, activation='identity')
        else:
            self.conv_last = CollapsibleLinearBlock(in_channels=num_channels,
                                                    out_channels=out_channels * scaling_factor ** 2,
                                                    tmp_channels=256, kernel_size=5, activation='identity')

        self.add_upsampled_input = AddOp()
        self.concat_upsampled_input = ConcatOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def forward(self, input):
        feature_shapes = []  # 레이어 이름과 feature map shape을 저장할 리스트

        # 입력 업샘플링
        upsampled_input = self.anchor(input)
        feature_shapes.append(('Anchor', upsampled_input.shape))  # 업샘플링된 입력의 shape 저장

        # conv-first 레이어
        initial_features = self.conv_first(input)
        feature_shapes.append(('Conv First', initial_features.shape))  # conv_first 이후의 shape 저장

        # Residual 블록과 CBAM 통과
        residual_features = initial_features
        for i, layer in enumerate(self.residual_layers):
            residual_features = layer(residual_features)  # 각 블록마다 CBAM 적용
            feature_shapes.append((f'Residual Block {i+1} with CBAM', residual_features.shape))

        # Residual 연결
        if self.use_concat_residual:
            residual_features = self.concat_residual(residual_features, initial_features)
            feature_shapes.append(('Concat Residual', residual_features.shape))  # Concat 후의 shape 저장
        else:
            residual_features = self.add_residual(residual_features, initial_features)
            feature_shapes.append(('Add Residual', residual_features.shape))  # Add 후의 shape 저장

        # conv-last 레이어
        final_features = self.conv_last(residual_features)
        feature_shapes.append(('Conv Last', final_features.shape))  # conv_last 이후의 shape 저장

        # 최종 출력
        output = self.add_upsampled_input(final_features, upsampled_input)

        if self.use_clipped_relu:
            output = self.depth_to_space(output)
            output = torch.clamp(output, min=0, max=1)
        else:
            output = self.depth_to_space(output)
        feature_shapes.append(('Final Output', output.shape))  # 최종 출력의 shape 저장

        return output, feature_shapes


class SESRRelease_M7(SESRRelease):
    def __init__(self, scaling_factor, clipped_relu:bool=False, concat_residual:bool=False, **kwargs):
        super(SESRRelease_M7, self).__init__(scaling_factor=scaling_factor, 
                                             num_channels=16, 
                                             num_lblocks=7, 
                                             use_clipped_relu=clipped_relu, 
                                             use_concat_residual=concat_residual, 
                                             **kwargs)


# SESRRelease_M7 모델 인스턴스 생성
model = SESRRelease_M7(scaling_factor=2, clipped_relu=False, concat_residual=False)

# 모델을 평가 모드로 전환
model.eval()

# 샘플 입력 (1개의 이미지, 3채널, 64x64 크기)
input_tensor = torch.randn(1, 3, 64, 64)

# 모델을 통과시켜 출력과 각 레이어에서의 feature map shape 획득
output, feature_shapes = model(input_tensor)

# 각 레이어에서의 feature map shape와 레이어 이름 출력
for i, (layer_name, shape) in enumerate(feature_shapes):
    print(f"Layer: {layer_name}, Feature Map {i+1} Shape: {shape}")