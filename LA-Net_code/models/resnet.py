#############原始，没加性别标签
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
#######定义注意力机制模块
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(SqueezeExcite, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 对3D输入使用Global Average Pooling

        # Excite: 两个全连接层
        self.fc1 = nn.Linear(in_channels, in_channels // self.ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // self.ratio, in_channels, bias=False)

    def forward(self, x):
        # Squeeze操作：全局平均池化
        z = self.avg_pool(x)  # 输出形状：[batch_size, channels, 1, 1, 1]

        # 将池化结果展平，并通过两个全连接层激活
        z = torch.flatten(z, 1)  # 展平为 [batch_size, channels]
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z)).view(-1, self.in_channels, 1, 1, 1)  # 恢复为 [batch_size, channels, 1, 1, 1]

        # Excite操作：与输入特征图相乘
        return x * z



# 3D卷积操作函数：3x3x3卷积
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes, # 输入通道数
        out_planes,# 输出通道数
        kernel_size=3,
        dilation=dilation,# 膨胀系数
        stride=stride,
        padding=dilation,
        bias=False)# 不使用偏置

# 用于下采样的基本模块：通过平均池化和零填充实现下采样
def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride) # 先做平均池化
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()# 零填充，补齐到期望的通道数
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))# 将池化结果和零填充合并

    return out


# 定义残差网络中的基础模块（BasicBlock）
class BasicBlock(nn.Module):
    expansion = 1 # 扩展比例，BasicBlock没有扩展

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, se_ratio=8):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)# 第一个3x3卷积层
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)# 第二个3x3卷积层
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        # 添加 Squeeze-Excite 模块
        # self.se = SqueezeExcite(planes, ratio=se_ratio)

    def forward(self, x):
        residual = x# 保存输入数据，以便于残差连接

        out = self.conv1(x)# 第一个卷积操作
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)# 第二个卷积操作
        out = self.bn2(out)

        # 通过 SE 模块
        # out = self.se(out)

        if self.downsample is not None:# 如果有下采样，则对输入做下采样
            residual = self.downsample(x)

        out += residual
        out = self.relu(out) # 最后再进行ReLU激活

        return out



# 定义瓶颈块（Bottleneck），用于深层网络（如ResNet50等）
class Bottleneck(nn.Module):
    expansion = 4# 扩展比例，瓶颈块的输出通道数是输入的4倍

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, se_ratio=8):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)# 第一个1x1卷积层
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)# 第二个3x3卷积层
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)# 第三个1x1卷积层
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)# 激活函数ReLU
        self.downsample = downsample# 下采样层
        self.stride = stride
        self.dilation = dilation

        # 添加 SE 模块
        # self.se = SqueezeExcite(planes * 4, ratio=se_ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)# 第一个卷积操作
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)# 第二个卷积操作
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)# 第三个卷积操作
        out = self.bn3(out)

        # 通过 SE 模块
        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)# 最后再进行ReLU激活

        return out



# 定义ResNet类
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64 # 初始输入通道数
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        # 输入层：7x7卷积
        self.conv1 = nn.Conv3d(
            1,# 输入通道数（灰度图像，1通道）
            64, # 输出通道数
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        # 批归一化
        self.bn1 = nn.BatchNorm3d(64)# 批量归一化
        self.relu = nn.ReLU(inplace=True) # 激活函数ReLU
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # 构建四个残差块（layer1 - layer4）
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, se_ratio=8)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, se_ratio=8)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2, se_ratio=8)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4, se_ratio=8)

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # 将每个特征图的空间维度降到1

        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, num_seg_classes)

        # self.conv_seg = nn.Sequential(
        #     nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(32, 1, kernel_size=1, stride=(1, 1, 1), bias=False)
        # )

        # self.fc_input_dim = self._get_fc_input_dim(1, sample_input_D, sample_input_H, sample_input_W)
        # self.fc = nn.Linear(self.fc_input_dim, num_seg_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_fc_input_dim(self, sample_input_C, sample_input_D, sample_input_H, sample_input_W):
        x = torch.zeros(1, sample_input_C, sample_input_D, sample_input_H, sample_input_W)
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = self.bn1(x)
        print(f"After bn1: {x.shape}")
        x = self.relu(x)
        print(f"After relu: {x.shape}")
        # x = self.maxpool(x)
        # print(f"After maxpool: {x.shape}")
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")
        x = self.layer2(x)
        print(f"After layer2: {x.shape}")
        x = self.layer3(x)
        print(f"After layer3: {x.shape}")
        x = self.layer4(x)
        print(f"After layer4: {x.shape}")

        # 添加全局平均池化
        x = self.global_avg_pool(x)
        print(f"After global_avg_pool: {x.shape}")
        # 展平特征图
        x = x.view(x.size(0), -1)
        print(f"Flattened shape: {x.shape}")
        return x.size(1)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1, se_ratio=8):
        downsample = None
        # 判断是否需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride,
                                     no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, se_ratio=se_ratio))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.conv_seg(x)

        # 全局平均池化
        x = self.global_avg_pool(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层输出
        x = self.fc(x)

        return x


# Set the appropriate input size (e.g., D=40, H=224, W=244)
# model = ResNet(BasicBlock, [1, 1, 1, 1], 40, 224, 244, num_seg_classes=1)  # Assuming 1 class for segmentation task
# print("FC input dimension:", model.fc_input_dim)


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model






#
# ############修改，加入性别标签
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import math
# from functools import partial
#
# __all__ = [
#     'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#     'resnet152', 'resnet200'
# ]
# #######定义注意力机制模块
# class SqueezeExcite(nn.Module):
#     def __init__(self, in_channels, ratio=8):
#         super(SqueezeExcite, self).__init__()
#         self.in_channels = in_channels
#         self.ratio = ratio
#
#         # Squeeze: Global Average Pooling
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 对3D输入使用Global Average Pooling
#
#         # Excite: 两个全连接层
#         self.fc1 = nn.Linear(in_channels, in_channels // self.ratio, bias=False)
#         self.fc2 = nn.Linear(in_channels // self.ratio, in_channels, bias=False)
#
#     def forward(self, x):
#         # Squeeze操作：全局平均池化
#         z = self.avg_pool(x)  # 输出形状：[batch_size, channels, 1, 1, 1]
#
#         # 将池化结果展平，并通过两个全连接层激活
#         z = torch.flatten(z, 1)  # 展平为 [batch_size, channels]
#         z = F.relu(self.fc1(z))
#         z = torch.sigmoid(self.fc2(z)).view(-1, self.in_channels, 1, 1, 1)  # 恢复为 [batch_size, channels, 1, 1, 1]
#
#         # Excite操作：与输入特征图相乘
#         return x * z
#
#
#
# # 3D卷积操作函数：3x3x3卷积
# def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
#     return nn.Conv3d(
#         in_planes, # 输入通道数
#         out_planes,# 输出通道数
#         kernel_size=3,
#         dilation=dilation,# 膨胀系数
#         stride=stride,
#         padding=dilation,
#         bias=False)# 不使用偏置
#
# # 用于下采样的基本模块：通过平均池化和零填充实现下采样
# def downsample_basic_block(x, planes, stride, no_cuda=False):
#     out = F.avg_pool3d(x, kernel_size=1, stride=stride) # 先做平均池化
#     zero_pads = torch.Tensor(
#         out.size(0), planes - out.size(1), out.size(2), out.size(3),
#         out.size(4)).zero_()# 零填充，补齐到期望的通道数
#     if not no_cuda:
#         if isinstance(out.data, torch.cuda.FloatTensor):
#             zero_pads = zero_pads.cuda()
#
#     out = Variable(torch.cat([out.data, zero_pads], dim=1))# 将池化结果和零填充合并
#
#     return out
#
#
# # 定义残差网络中的基础模块（BasicBlock）
# class BasicBlock(nn.Module):
#     expansion = 1 # 扩展比例，BasicBlock没有扩展
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, se_ratio=8):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)# 第一个3x3卷积层
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes, dilation=dilation)# 第二个3x3卷积层
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation
#
#         # 添加 Squeeze-Excite 模块
#         # self.se = SqueezeExcite(planes, ratio=se_ratio)
#
#     def forward(self, x):
#         residual = x# 保存输入数据，以便于残差连接
#
#         out = self.conv1(x)# 第一个卷积操作
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)# 第二个卷积操作
#         out = self.bn2(out)
#
#         # 通过 SE 模块
#         # out = self.se(out)
#
#         if self.downsample is not None:# 如果有下采样，则对输入做下采样
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out) # 最后再进行ReLU激活
#
#         return out
#
#
#
# # 定义瓶颈块（Bottleneck），用于深层网络（如ResNet50等）
# class Bottleneck(nn.Module):
#     expansion = 4# 扩展比例，瓶颈块的输出通道数是输入的4倍
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, se_ratio=8):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)# 第一个1x1卷积层
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)# 第二个3x3卷积层
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)# 第三个1x1卷积层
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)# 激活函数ReLU
#         self.downsample = downsample# 下采样层
#         self.stride = stride
#         self.dilation = dilation
#
#         # 添加 SE 模块
#         # self.se = SqueezeExcite(planes * 4, ratio=se_ratio)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)# 第一个卷积操作
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)# 第二个卷积操作
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)# 第三个卷积操作
#         out = self.bn3(out)
#
#         # 通过 SE 模块
#         # out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)# 最后再进行ReLU激活
#
#         return out
#
#
#
# # 定义ResNet类
# class ResNet(nn.Module):
#     def __init__(self,
#                  block,
#                  layers,
#                  sample_input_D,
#                  sample_input_H,
#                  sample_input_W,
#                  num_seg_classes,
#                  shortcut_type='B',
#                  no_cuda=False):
#         self.inplanes = 64 # 初始输入通道数
#         self.no_cuda = no_cuda
#         super(ResNet, self).__init__()
#         # 输入层：7x7卷积
#         self.conv1 = nn.Conv3d(
#             1,# 输入通道数（灰度图像，1通道）
#             64, # 输出通道数
#             kernel_size=7,
#             stride=(2, 2, 2),
#             padding=(3, 3, 3),
#             bias=False)
#         # 批归一化
#         self.bn1 = nn.BatchNorm3d(64)# 批量归一化
#         self.relu = nn.ReLU(inplace=True) # 激活函数ReLU
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#
#         # 构建四个残差块（layer1 - layer4）
#         self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, se_ratio=8)
#         self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, se_ratio=8)
#         self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2, se_ratio=8)
#         self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4, se_ratio=8)
#
#         # 全局平均池化层
#         self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # 将每个特征图的空间维度降到1
#         # 性别标签全连接层
#         self.fc_sex = nn.Linear(1, 1)  # 如果性别是一个二分类的 0 或 1，可以用一个线性层来映射
#         # 全连接层
#         self.fc = nn.Linear(512 * block.expansion+1, num_seg_classes)
#
#         # self.conv_seg = nn.Sequential(
#         #     nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
#         #     nn.BatchNorm3d(32),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
#         #     nn.BatchNorm3d(32),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv3d(32, 1, kernel_size=1, stride=(1, 1, 1), bias=False)
#         # )
#
#         # self.fc_input_dim = self._get_fc_input_dim(1, sample_input_D, sample_input_H, sample_input_W)
#         # self.fc = nn.Linear(self.fc_input_dim, num_seg_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _get_fc_input_dim(self, sample_input_C, sample_input_D, sample_input_H, sample_input_W):
#         x = torch.zeros(1, sample_input_C, sample_input_D, sample_input_H, sample_input_W)
#         print(f"Input shape: {x.shape}")
#         x = self.conv1(x)
#         print(f"After conv1: {x.shape}")
#         x = self.bn1(x)
#         print(f"After bn1: {x.shape}")
#         x = self.relu(x)
#         print(f"After relu: {x.shape}")
#         # x = self.maxpool(x)
#         # print(f"After maxpool: {x.shape}")
#         x = self.layer1(x)
#         print(f"After layer1: {x.shape}")
#         x = self.layer2(x)
#         print(f"After layer2: {x.shape}")
#         x = self.layer3(x)
#         print(f"After layer3: {x.shape}")
#         x = self.layer4(x)
#         print(f"After layer4: {x.shape}")
#
#         # 添加全局平均池化
#         x = self.global_avg_pool(x)
#         print(f"After global_avg_pool: {x.shape}")
#         # 展平特征图
#         x = x.view(x.size(0), -1)
#         print(f"Flattened shape: {x.shape}")
#         return x.size(1)
#
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1, se_ratio=8):
#         downsample = None
#         # 判断是否需要下采样
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride,
#                                      no_cuda=self.no_cuda)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm3d(planes * block.expansion))
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, se_ratio=se_ratio))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, volume, sex_label):
#         # 处理 volume 输入
#         x = self.conv1(volume)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)  # 如果有最大池化层，请取消注释
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         # 全局平均池化
#         x = self.global_avg_pool(x)
#
#         # 展平特征图
#         x = x.view(x.size(0), -1)  # x 现在是 [batch_size, features]
#
#         # 打印 x 的形状
#         # print(f"Shape of x before concatenation: {x.shape}")
#
#         # 检查并调整 sex_label 的形状
#         if sex_label.dim() == 1:
#             # 假设 sex_label 是一个一维张量，包含每个样本的性别标签
#             if sex_label.size(0) != x.size(0):
#                 raise ValueError(f"Batch sizes do not match: x has {x.size(0)}, but sex_label has {sex_label.size(0)}")
#             sex_label = sex_label.reshape(-1, 1).float()  # 转换为 [batch_size, 1]
#         elif sex_label.dim() == 2 and sex_label.shape[1] == 1:
#             # 如果已经是 [batch_size, 1] 的形状，不需要额外操作
#             if sex_label.size(0) != x.size(0):
#                 raise ValueError(f"Batch sizes do not match: x has {x.size(0)}, but sex_label has {sex_label.size(0)}")
#         else:
#             raise ValueError(
#                 f"Unexpected shape for sex_label: {sex_label.shape}. Expected either [batch_size] or [batch_size, 1]")
#
#         # 打印调试信息（查看 x 和 sex_label 的形状）
#         # print(f"Shape of x: {x.shape}, Shape of sex_label: {sex_label.shape}")
#
#         # 拼接 x 和 sex_label，在特征维度上
#         x = torch.cat((x, sex_label), dim=1)  # 在特征维度拼接
#
#         # 打印拼接后的形状
#         # print(f"Shape after concatenation: {x.shape}")
#
#         # 最终分类层
#         x = self.fc(x)
#
#         return x
#
# # Set the appropriate input size (e.g., D=40, H=224, W=244)
# # model = ResNet(BasicBlock, [1, 1, 1, 1], 40, 224, 244, num_seg_classes=1)  # Assuming 1 class for segmentation task
# # print("FC input dimension:", model.fc_input_dim)
#
#
# def resnet10(**kwargs):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
#     return model
#
#
# def resnet18(**kwargs):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model
#
#
# def resnet34(**kwargs):
#     """Constructs a ResNet-34 model.
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet50(**kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet101(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     return model
#
#
# def resnet152(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     return model
#
#
# def resnet200(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
#     return model





# ##########多任务
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import math
# from functools import partial
#
# __all__ = [
#     'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#     'resnet152', 'resnet200'
# ]
# #######定义注意力机制模块
# class SqueezeExcite(nn.Module):
#     def __init__(self, in_channels, ratio=8):
#         super(SqueezeExcite, self).__init__()
#         self.in_channels = in_channels
#         self.ratio = ratio
#
#         # Squeeze: Global Average Pooling
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 对3D输入使用Global Average Pooling
#
#         # Excite: 两个全连接层
#         self.fc1 = nn.Linear(in_channels, in_channels // self.ratio, bias=False)
#         self.fc2 = nn.Linear(in_channels // self.ratio, in_channels, bias=False)
#
#     def forward(self, x):
#         # Squeeze操作：全局平均池化
#         z = self.avg_pool(x)  # 输出形状：[batch_size, channels, 1, 1, 1]
#
#         # 将池化结果展平，并通过两个全连接层激活
#         z = torch.flatten(z, 1)  # 展平为 [batch_size, channels]
#         z = F.relu(self.fc1(z))
#         z = torch.sigmoid(self.fc2(z)).view(-1, self.in_channels, 1, 1, 1)  # 恢复为 [batch_size, channels, 1, 1, 1]
#
#         # Excite操作：与输入特征图相乘
#         return x * z
#
#
#
# # 3D卷积操作函数：3x3x3卷积
# def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
#     return nn.Conv3d(
#         in_planes, # 输入通道数
#         out_planes,# 输出通道数
#         kernel_size=3,
#         dilation=dilation,# 膨胀系数
#         stride=stride,
#         padding=dilation,
#         bias=False)# 不使用偏置
#
# # 用于下采样的基本模块：通过平均池化和零填充实现下采样
# def downsample_basic_block(x, planes, stride, no_cuda=False):
#     out = F.avg_pool3d(x, kernel_size=1, stride=stride) # 先做平均池化
#     zero_pads = torch.Tensor(
#         out.size(0), planes - out.size(1), out.size(2), out.size(3),
#         out.size(4)).zero_()# 零填充，补齐到期望的通道数
#     if not no_cuda:
#         if isinstance(out.data, torch.cuda.FloatTensor):
#             zero_pads = zero_pads.cuda()
#
#     out = Variable(torch.cat([out.data, zero_pads], dim=1))# 将池化结果和零填充合并
#
#     return out
#
#
# # 定义残差网络中的基础模块（BasicBlock）
# class BasicBlock(nn.Module):
#     expansion = 1 # 扩展比例，BasicBlock没有扩展
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, se_ratio=8):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)# 第一个3x3卷积层
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes, dilation=dilation)# 第二个3x3卷积层
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation
#
#         # 添加 Squeeze-Excite 模块
#         # self.se = SqueezeExcite(planes, ratio=se_ratio)
#
#     def forward(self, x):
#         residual = x# 保存输入数据，以便于残差连接
#
#         out = self.conv1(x)# 第一个卷积操作
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)# 第二个卷积操作
#         out = self.bn2(out)
#
#         # 通过 SE 模块
#         # out = self.se(out)
#
#         if self.downsample is not None:# 如果有下采样，则对输入做下采样
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out) # 最后再进行ReLU激活
#
#         return out
#
#
#
# # 定义瓶颈块（Bottleneck），用于深层网络（如ResNet50等）
# class Bottleneck(nn.Module):
#     expansion = 4# 扩展比例，瓶颈块的输出通道数是输入的4倍
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, se_ratio=8):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)# 第一个1x1卷积层
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)# 第二个3x3卷积层
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)# 第三个1x1卷积层
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)# 激活函数ReLU
#         self.downsample = downsample# 下采样层
#         self.stride = stride
#         self.dilation = dilation
#
#         # 添加 SE 模块
#         # self.se = SqueezeExcite(planes * 4, ratio=se_ratio)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)# 第一个卷积操作
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)# 第二个卷积操作
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)# 第三个卷积操作
#         out = self.bn3(out)
#
#         # 通过 SE 模块
#         # out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)# 最后再进行ReLU激活
#
#         return out
#
#
#
# # 定义ResNet类
#
# class ResNet(nn.Module):
#     def __init__(self,
#                  block,
#                  layers,
#                  sample_input_D,
#                  sample_input_H,
#                  sample_input_W,
#                  num_tasks=4,  # 任务数量（年龄 + 3 个肺功能指标）
#                  shortcut_type='B',
#                  no_cuda=False):
#         self.inplanes = 64
#         self.no_cuda = no_cuda
#         super(ResNet, self).__init__()
#
#         self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#
#         self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
#         self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
#
#         self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
#
#         self.fc_sex = nn.Linear(1, 1)
#
#         self.fc_shared = nn.Linear(512 * block.expansion + 1, 256)  # 先降维
#         self.fc_age = nn.Linear(256, 1)  # 预测年龄
#         self.fc_fvc = nn.Linear(256, 1)  # 预测校正 FVC
#         self.fc_fev1 = nn.Linear(256, 1)  # 预测校正 FEV1
#         self.fc_fev1_ratio = nn.Linear(256, 1)  # 预测 FEV1/FEV1_Pred
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(downsample_basic_block,
#                                      planes=planes * block.expansion,
#                                      stride=stride,
#                                      no_cuda=self.no_cuda)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(self.inplanes, planes * block.expansion,
#                               kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm3d(planes * block.expansion),
#                 )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, volume, sex_label):
#         x = self.conv1(volume)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.global_avg_pool(x)
#         x = x.view(x.size(0), -1)
#
#         if sex_label.dim() == 1:
#             sex_label = sex_label.view(-1, 1).float()
#         elif sex_label.dim() == 2 and sex_label.shape[1] == 1:
#             pass
#         else:
#             raise ValueError(f"Unexpected shape for sex_label: {sex_label.shape}")
#
#         x = torch.cat((x, sex_label), dim=1)
#
#         x = F.relu(self.fc_shared(x))
#
#         age_pred = self.fc_age(x)
#         fvc_pred = self.fc_fvc(x)
#         fev1_pred = self.fc_fev1(x)
#         fev1_ratio_pred = self.fc_fev1_ratio(x)
#
#         return age_pred, fvc_pred, fev1_pred, fev1_ratio_pred
# # Set the appropriate input size (e.g., D=40, H=224, W=244)
# # model = ResNet(BasicBlock, [1, 1, 1, 1], 40, 224, 244, num_seg_classes=1)  # Assuming 1 class for segmentation task
# # print("FC input dimension:", model.fc_input_dim)
#
#
# def resnet10(**kwargs):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
#     return model
#
#
# def resnet18(**kwargs):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model
#
#
# def resnet34(**kwargs):
#     """Constructs a ResNet-34 model.
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet50(**kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet101(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     return model
#
#
# def resnet152(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     return model
#
#
# def resnet200(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
#     return model