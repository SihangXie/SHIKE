# -*- coding:utf-8 -*-
"""
@Project: SHIKE
@File：dis_align_linear.py
@Author：Sihang Xie
@Time：2024/1/23 11:08
@Description：DisAlign Linear Module
"""

import math
import torch
from torch.nn import functional as F
from torch.functional import Tensor


class NormalizedLinear(torch.nn.Module):
    """
    An advanced Linear layer which supports weight normalization or cosine normalization.

    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            feat_norm=True,
            scale_mode='learn',
            scale_init=1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.scale_mode = scale_mode
        self.scale_init = scale_init

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if self.scale_mode == 'constant':
            self.scale = scale_init
        elif self.scale_mode == 'learn':
            self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): (N, C)
        Return:
            output (torch.Tensor): (N, D)
        """
        if self.feat_norm:
            inputs = F.normalize(inputs, dim=1)

        output = inputs.mm(F.normalize(self.weight, dim=1).t())
        output = self.scale * output
        return output

    def extra_repr(self):
        s = ('in_features={in_features}, out_features={out_features}')
        if self.bias is None:
            s += ', bias=False'
        s += ', feat_norm={feat_norm}'
        s += ', scale_mode={scale_mode}'
        s += ', scale_init={scale_init}'

        return s.format(**self.__dict__)


class DisAlignLinear(torch.nn.Linear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    1.阶段一可学习参数是否需要更新？还是阶段二才开始更新？
    2.阶段一是否需要进行校准？
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: Tensor):
        logit_before = F.linear(input, self.weight, self.bias)  # 对原始分类器进行原地校准
        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
                      confidence * self.logit_bias
        return logit_after


class DisAlignNormalizedLinear(NormalizedLinear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, **args) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **args)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: Tensor):
        if self.feat_norm:
            input = F.normalize(input, dim=1)

        output = input.mm(F.normalize(self.weight, dim=1).t())
        logit_before = self.scale * output

        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
                      confidence * self.logit_bias
        return logit_after
