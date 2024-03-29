import copy
from typing import *
import torch.nn as nn
from torch.nn import functional as F
import torch


class SpareBinaryQuanter:
    def __int__(self):
        self.QuantedModelStateDict:OrderedDict[str, torch.Tensor] = None
        self.DeQuantedModelStateDict:OrderedDict[str, torch.Tensor] = None
        self.QuantMapDict: Dict[str, float] = None

    def find_top_positive(self):

    def quant_model(self, mode_state_dict):
        pass


    def dequant_model(self):
        pass


    def quant(self, x, *y):
        pass


    def dequant(self, x, *y):
        pass

class QuantModelDict:
    def __init__(self):
        self.QuantedModelStateDict = None
        self.DeQuantedModelStateDict = None

        self.scale = 0
        self.zero_point = 0

        self.bit = 8
        self.q_max = 2 ** (self.bit-1) - 1
        self.q_min = -1 * 2 ** (self.bit-1)


        self.layers_name = []
        self.layers_scale_zero_point = {}

    def set_compression_bits(self, bits):

        # 设置要压缩的位数
        self.bit = bits
        self.q_max = 2 ** (self.bit - 1) - 1
        self.q_min = -1 * 2 ** (self.bit - 1)

    def init_layers(self, model:nn.Module):
        for name, module in model.named_parameters():
            self.layers_name.append(name)
            self.layers_scale_zero_point[name] = [0, 0]


    def quant(self, r, *y):
        return round(r / self.scale + self.zero_point)

    def quant_test(self, r, *y):
        q = round(r / self.scale + self.zero_point)
        return  q

    def dequant(self, q, *y):
        res = self.scale * (q - self.zero_point)
        return res

    def quant_model(self, model_state_dict):
        # 将原始模型字典进行权重量化
        self.QuantedModelStateDict = copy.deepcopy(model_state_dict)
        for name, param in model_state_dict.items():
            max_value = torch.max(param, ).item()
            min_value = torch.min(param, ).item()

            self.scale = (max_value - min_value) / (self.q_max - self.q_min)
            self.zero_point = round(self.q_max - max_value / self.scale)
            new_param = param.clone()

            self.layers_scale_zero_point[name] = [self.scale, self.zero_point]
            quanted_param = new_param.map_(new_param, self.quant_test).to(torch.int8)
            self.QuantedModelStateDict[name] = quanted_param
        return self.QuantedModelStateDict

    def quant_model_test(self, model_state_dict):
        # 将原始模型字典进行权重量化
        self.QuantedModelStateDict = copy.deepcopy(model_state_dict)
        for name, param in model_state_dict.items():
            max_value = torch.max(param, ).item()
            min_value = torch.min(param, ).item()

            self.scale = (max_value - min_value) / (self.q_max - self.q_min)
            self.zero_point = round(self.q_max - max_value / self.scale)
            new_param = param.clone()

            self.layers_scale_zero_point[name] = [self.scale, self.zero_point]
            quanted_param = new_param.map_(new_param, self.quant_test).to(torch.int8)
            self.QuantedModelStateDict[name] = quanted_param
        return self.QuantedModelStateDict

    def dequant_model(self, model_state_dict):

        if self.QuantedModelStateDict is not None:
            self.DeQuantedModelStateDict = copy.deepcopy(model_state_dict)

            for name, param in self.QuantedModelStateDict.items():
                self.scale, self.zero_point = self.layers_scale_zero_point[name]

                new_param = param.clone().to(torch.float32)

                dequanted_param = new_param.map_(new_param, self.dequant)
                # print(param - dequanted_param)
                self.DeQuantedModelStateDict[name] = dequanted_param
        return self.DeQuantedModelStateDict

class BaseQuantModel(nn.Module):
    def __init__(self):
        super(BaseQuantModel, self).__init__()
        self.Quanter = QuantModelDict()

    def quant(self):
        # 量化模型权重 返回模型字典
        return self.Quanter.quant_model(self.state_dict())

    def dequant(self):
        return self.Quanter.dequant_model(self.state_dict())

    def quant_test(self):
        return self.Quanter.quant_model_test(self.state_dict())



class Cifar10Model(BaseQuantModel):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 10)
        self.Quanter.init_layers(self)

    def share_memory(self):
        for param in self.parameters():
            param.share_memory_()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.log_softmax(self.fc4(x), dim=1)


class NewCifar10Model(BaseQuantModel):
    def __init__(self):
        super(NewCifar10Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5))
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self.Quanter.init_layers(self)

    def share_memory(self):
        for param in self.parameters():
            param.share_memory_()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(self.fc3(x), dim=1)


class MnistModel(BaseQuantModel):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 10)

        self.Quanter.init_layers(self)

    def share_memory(self):
        for param in self.parameters():
            param.share_memory_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)


class AlexNetCIFAR(BaseQuantModel):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)  # 使用log_softmax作为激活函数
        return x

    def share_memory(self):
        for param in self.parameters():
            param.share_memory_()

class SimpleLinear(BaseQuantModel):

    def __init__(self, h1=2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def share_memory(self):
        for param in self.parameters():
            param.share_memory_()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def init_model(model_name):
    if model_name == 'mnist':
        return MnistModel()
    elif model_name == 'cifar10':
        return Cifar10Model()
    elif model_name == "NewCifar10":
        return NewCifar10Model()
    elif model_name == "simple_mnist":
        return SimpleLinear()
    elif model_name == "AlexNetCIFAR":
        return AlexNetCIFAR()
    else:
        return None

