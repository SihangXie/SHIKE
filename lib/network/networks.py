'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
This MoE design is based on the implementation of Yerlan Idelbayev.
'''

from collections import OrderedDict
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch.cuda.amp import autocast as autocast, GradScaler


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StridedConv(nn.Module):
    """
    downsampling conv layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out


class ShallowExpert(nn.Module):
    """
    shallow features alignment wrt. depth
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(ShallowExpert, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}',
                          StridedConv(in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)),
                                      use_relu=(k != 1))) for
                         k in range(depth)]))

    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class ResNet_MoE(nn.Module):

    def __init__(self, block, num_blocks, num_experts=None, num_classes=10, use_norm=False):
        super(ResNet_MoE, self).__init__()
        self.s = 1  # 共享层数？
        self.num_experts = num_experts
        self.in_planes = 16
        self.next_in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # 第一层深度共享
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # 第二层深度共享

        if num_experts:  # 第3个阶段是每个专家专属的特征
            layer3_output_dim = 64
            self.in_planes = 32
            self.layer3s = nn.ModuleList([self._make_layer(  # 为3个专家分别构建3个layer3
                block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(self.num_experts)])
            self.in_planes = self.next_in_planes
            if use_norm:
                self.s = 30  # 所以这个s是什么？
                self.classifiers = nn.ModuleList(
                    [NormedLinear(64, num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(64, num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList(  # 为3个专家分别构建3个分类头
                    [nn.Linear(64, num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 不使用多专家只构建1个layer3和1个分类器
            self.linear = NormedLinear(64, num_classes) if use_norm else nn.Linear(
                64, num_classes, bias=True)

        self.apply(_weights_init)  # 递归地对模块进行应用初始化
        self.depth = list(  # 定义网络深度
            reversed([i + 1 for i in range(len(num_blocks) - 1)]))  # [2, 1]
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(  # 定义专家分配到的网络深度
            self.num_experts)]  # [2, 1 , 2]
        feat_dim = 16
        self.shallow_exps = nn.ModuleList([ShallowExpert(  # 构建浅层的专家的对齐下采样层
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])

        self.shallow_avgpool = nn.AdaptiveAvgPool2d((8, 8))  # 自适应平均池化成形状为[bs, c, 8, 8]的张量

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x, crt=False):  # 入参crt表示是否分类器训练

        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)  # 第1个共享特征图[bs, 16, 32, 32]

        out2 = self.layer2(out1)  # 第2个共享特征图[bs, 32, 16, 16]
        shallow_outs = [out1, out2]  # 2个不同深度的共享浅层特征
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]  # 3个专家的专属层会形成一个输出列表，值都不一样
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]  # 特征图形状对齐

            exp_outs = [out3s[i] * shallow_expe_outs[i]  # 对齐后的浅层共享特征与专家专属特征进行哈达玛积融合
                        for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(  # 全局平均池化(bs, 64, 1, 1)
                output.size(0), -1) for output in exp_outs]  # 然后拉平(bs, 64)
            embeddings = exp_outs

            if crt == True:  # 分类器训练
                outs = [self.s * self.rt_classifiers[i]
                (embeddings[i]) for i in range(self.num_experts)]
            else:  # 特征提取网络训练
                outs = [self.s * self.classifiers[i]  # 为什么要点乘`self.s`？分类器是`NormedLinear`
                (embeddings[i]) for i in range(self.num_experts)]  # 分类器输出(bs, 100)
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            embeddings = out
            outs = self.linear(out)

        return outs, embeddings

    def load_model(self, model_path, **kwargs):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if 'backbone_only' in kwargs.keys() and 'classifier' in k:
                continue;
            if k.startswith("module"):
                if k[7:] not in model_dict.keys():
                    print('not load:{}'.format(k))
                    continue
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")


# for Cifar100-LT use


def resnet32(num_classes=100, use_norm=False, num_exps=None):
    return ResNet_MoE(BasicBlock, [5, 5, 5], num_experts=num_exps, num_classes=num_classes, use_norm=use_norm)


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params:", total_params)
    print("Total layers:", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    moe32 = resnet32(num_classes=100, num_exps=3, use_norm=False)
    test(net=moe32)
