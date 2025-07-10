# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn
import torch

class GroupedLinear(nn.Module):
    """
    입력 채널(in_features)을 여러 그룹(num_groups)으로 나누어
    독립된 선형 변환 후 합치는 GroupedLinear 레이어.
    내부적으로 torch.bmm를 사용해 병렬 처리.
    """
    def __init__(self, in_features, out_features, num_groups, bias=True):
        super().__init__()
        assert in_features % num_groups == 0, \
            "in_features must be divisible by num_groups."
        assert out_features % num_groups == 0, \
            "out_features must be divisible by num_groups."

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_in = in_features // num_groups
        self.group_out = out_features // num_groups

        # (num_groups, group_in, group_out) 형태의 가중치
        self.weight = nn.Parameter(
            torch.Tensor(num_groups, self.group_in, self.group_out)
        )
        # 바이어스
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_groups, self.group_out))
        else:
            self.register_parameter('bias', None)

        # 간단히 Xavier 초기화
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, ..., in_features)
           - 2D 또는 3D 이상 텐서도 뒷차원(in_features)을 변환한다고 가정 가능
        """
        # x의 마지막 차원이 in_features라 가정
        orig_shape = x.shape[:-1]          # (batch_size, seq_len, ...)
        in_dim = x.shape[-1]              # in_features
        
        # 먼저 2D 형태로 (B', in_features)만 떼어내기
        x_2d = x.reshape(-1, in_dim)      # (batch_size * seq_len, in_features)
        B = x_2d.size(0)                  # batch_size * seq_len

        # (B, in_features) -> (B, num_groups, group_in)
        x_2d = x_2d.view(B, self.num_groups, self.group_in)

        # bmm를 위해 (num_groups, B, group_in)으로 차원 변경
        x_2d = x_2d.permute(1, 0, 2)  # (num_groups, B, group_in)
        
        # weight: (num_groups, group_in, group_out)
        # x_2d:   (num_groups, B, group_in)
        # bmm -> (num_groups, B, group_out)
        out = torch.bmm(x_2d, self.weight)

        # (num_groups, B, group_out) -> (B, num_groups, group_out)
        out = out.permute(1, 0, 2)

        # 바이어스 적용
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)  # (1, num_groups, group_out)

        # (B, num_groups * group_out) = (B, out_features)
        out = out.reshape(B, -1)

        # 원래 배치 차원으로 복원
        out = out.view(*orig_shape, self.out_features)
        return out
    
# import torch.nn.functional as F
# class GroupedLinear(nn.Module):
#     def __init__(self, in_features, out_features, num_groups, bias=True,
#                  use_half=False):
#         super().__init__()
#         assert in_features % num_groups == 0
#         assert out_features % num_groups == 0

#         self.in_features  = in_features
#         self.out_features = out_features
#         self.num_groups   = num_groups
#         self.group_in     = in_features  // num_groups
#         self.group_out    = out_features // num_groups
#         self.use_half     = use_half     # fp16 or fp32

#         # conv1d weight shape: (out_ch, in_ch/groups, k=1)
#         w = torch.empty(out_features, self.group_in, 1)
#         nn.init.xavier_uniform_(w)
#         self.weight = nn.Parameter(w)

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features))
#         else:
#             self.register_parameter("bias", None)

#     def forward(self, x):
#         orig = x.shape[:-1]                 # (B, …)
#         x = x.reshape(-1, self.in_features, 1)   # (B', C_in, L=1)

#         # ⚠️ channels_last 제거 — Conv1d에는 의미 없음
#         with torch.autocast('cuda', enabled=self.use_half):
#             y = F.conv1d(
#                 x, self.weight, self.bias,
#                 groups=self.num_groups
#             ).squeeze(-1)                   # (B', out_features)

#         return y.reshape(*orig, self.out_features)
    
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Grouped_Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = GroupedLinear(
            in_features=in_features,
            out_features=hidden_features,
            num_groups=num_groups,     
            bias=bias
        )
        
        self.act = act_layer()
        self.fc2 =  GroupedLinear(
            in_features=hidden_features,
            out_features=out_features,
            num_groups=num_groups,     
            bias=bias
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x