# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Important stuff for the C2F module.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.global_vars as gvars
import numpy as np
from .DA import _ImageDA


class C2FMaskHead(nn.Module):
    def __init__(self):
        """
        Coarse to fine Mask Head.
        # --CFF Module
        """
        super().__init__()

        # self.in_features = ['C2', 'T5']
        self.num_channels = 512
        self.out_channels = 256

        self.num_levels = 2
        self.max_pooling_8 = nn.MaxPool2d(kernel_size=3, stride=8, padding=1)

        self.multihead_attn = nn.MultiheadAttention(self.out_channels, 8, dropout=0.1)

        # 3x3 Convolution followed by BatchNorm and ReLU
        self.inner_seq = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels+1, out_channels=self.num_channels, kernel_size=1),
            # nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
            # nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1,
            #           padding=1),
        )

        # # 3x3 conv
        # self.conv2d = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        # 4 max pool
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        # 1x1 conv
        self.conv1x1 = nn.Conv2d(self.num_channels, self.out_channels, kernel_size=1)


    def reset_parameters_(module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()


    def forward(self, features, edge_features):
        assert len(features) == self.num_levels, print("The number of input features should be equal to the supposed level.")

        high_level_features = features[0] # ([2, 256, 24, 48])
        low_level_features = features[1] # ([2, 512, 96, 192])
        edge_features_small = self.max_pooling_8(edge_features.float())

        # normalized_tensor1 = (low_level_features - low_level_features.min()) / (low_level_features.max() - low_level_features.min())
        # normalized_tensor2 = (edge_features_small - edge_features_small.min()) / (edge_features_small.max() - edge_features_small.min())

        # breakpoint()
        # Sum the tensors
        concatenated_tensors = torch.cat((low_level_features, edge_features_small), dim=1)
        # concatenated_tensors = low_level_features

        # Inner sequential
        out_inner = self.inner_seq(concatenated_tensors)

        # Multiply the original tensors with the sigmoid output
        fused = low_level_features * out_inner

        downscaled_fusion = self.max_pool(self.conv1x1(fused))
        # breakpoint()

        fusion_flattened = downscaled_fusion.flatten(2).permute(2, 0, 1)
        high_level_flattened = high_level_features.flatten(2).permute(2, 0, 1)
        b, c, h, w = high_level_features.shape
        # attention_result Size [H*W, Batch, Channels]
        attention_result = self.multihead_attn(query=high_level_flattened, key=fusion_flattened, value=fusion_flattened)
        restored_result = attention_result[0].permute(1, 2, 0).view((b,c,h,w))

        # Define the target size
        target_size = (12, 25)
        # Resize the tensor
        resized_attention = F.interpolate(restored_result, size=target_size, mode='bilinear', align_corners=False).flatten(2).permute(0, 2, 1)


        if gvars.print_plots_C2f:
            fig, axs = plt.subplots(3, 3, figsize=(15, 10))

            axs[0, 0].imshow(torch.mean(low_level_features[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[0, 0].title.set_text('low_level_features')

            axs[0, 1].imshow(torch.mean(edge_features_small[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[0, 1].title.set_text('edge_features_small')


            axs[1, 0].imshow(torch.mean(high_level_features[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[1, 0].title.set_text('high_level_features')


            axs[1, 1].imshow(torch.mean(fused[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[1, 1].title.set_text('fused')

            axs[2, 0].imshow(torch.mean(downscaled_fusion[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[2, 0].title.set_text('downscaled_fusion')

            axs[2, 1].imshow(torch.mean(attention_result[0].permute(1, 2, 0).view((b,c,h,w))[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[2, 1].title.set_text('attention_result')


            axs[0, 2].imshow(torch.mean(concatenated_tensors[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[0, 2].title.set_text('low+edge2')


            # axs[1, 2].imshow(torch.mean(normalized_tensor1[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[1, 2].hist(torch.mean(low_level_features[0], axis=0).cpu().detach().numpy(), bins=30, density=True, alpha=0.7, color=np.random.rand(176, 3))
            axs[1, 2].title.set_text('normalized_tensor1')


            # axs[2, 2].imshow(torch.mean(normalized_tensor2[0], axis=0).cpu().detach().numpy(), cmap='viridis')
            axs[2, 2].hist(torch.mean(edge_features_small[0], axis=0).cpu().detach().numpy(), bins=30, density=True, alpha=0.7, color=np.random.rand(176, 3))
            axs[2, 2].title.set_text('normalized_tensor2')

            plt.savefig('C2F_plots.png')
            breakpoint()
        return resized_attention

