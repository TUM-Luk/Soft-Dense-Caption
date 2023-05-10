import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from .blocks import MLP, ResidualBlock, UBlock


class ProposalModule(nn.Module):
    def __init__(self, in_channels=32, out_channels=6, norm_fn=None, num_layers=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            # nn.BatchNorm1d(num_features=in_channels,eps=1e-4, momentum=0.1),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

    def forward(self, data_dict):
        # bbox_features为detection module生成的instance features
        # pred_bboxes为预测出的bbox参数（中心和三维）
        data_dict["pred_bboxes"] = self.fc(data_dict['select_feats'])

        return data_dict

# test
# test_mlp = ProposalModule()
# print(test_mlp)
# test_mlp.eval()
# input=torch.rand(1,500,32)
# output = test_mlp.fc(input)
# print(output)
# print(output.shape)