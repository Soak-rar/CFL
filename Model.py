import copy
from typing import *

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
import math

import Args

args = Args.Arguments()


class STPandNQuanter:
    def __init__(self):
        self.QuantedModelStateDict: OrderedDict[str, torch.Tensor] = None
        self.DeQuantedModelStateDict: OrderedDict[str, torch.Tensor] = None

        self.layers_scale_zero_point = {}  # str: [quant_value, bool] true = pos, false = neg
        self.spare_rate = 0.05

        self.quant_value = [0, 0]
        self.symbol = None
        self.edge_value = [0, 0]

    def set_spare_rate(self, new_spare_rate):
        self.spare_rate = new_spare_rate

    def find_top_positive(self):
        pass

    def init_layers(self, model: nn.Module):
        for name, module in model.named_parameters():
            self.layers_scale_zero_point[name] = [0, None]

    def quant_layer(self, layer_param):
        mean_values, self.edge_value = self.get_layer_quant_value(copy.deepcopy(layer_param))
        # print(quant_value, _, self.edge_value)
        self.quant_value = mean_values

        new_param = layer_param.clone()
        quanted_param = new_param.map_(new_param, self.quant)
        return quanted_param

    def quant_model(self, model_state_dict):
        self.QuantedModelStateDict = copy.deepcopy(model_state_dict)
        for name, param in model_state_dict.items():

            if name == args.deep_model_layer_name:
                continue
            self.QuantedModelStateDict[name] = self.quant_layer(param)
        return self.QuantedModelStateDict

    #  非对称的 top_k
    def get_layer_quant_value(self, layer_tensor):
        # 计算每层的 稀疏量化值
        cat_tensor = layer_tensor.reshape(-1)
        top_size = math.ceil(cat_tensor.size()[0] * self.spare_rate)

        abs_cat_tensor = torch.abs(cat_tensor)

        abs_top_values = torch.topk(abs_cat_tensor, top_size, largest=True)

        top_tensor = torch.take(cat_tensor, abs_top_values[1])

        pos_list = []
        neg_list = []

        for v in top_tensor:
            if v > 0:
                pos_list.append(v)
            elif v < 0:
                neg_list.append(v)

        pos_mean = np.mean(pos_list)
        neg_mean = np.mean(neg_list)

        # print('------------------')
        # print(pos_top_values)
        #
        # print(neg_top_values)
        # print('------------------')
        mean_v = torch.mean(abs_top_values[0])
        return [mean_v, mean_v], [min(pos_list), max(neg_list)]

    def map_pos(self, x, *y):
        if x > 0:
            return x
        return 0

    def map_neg(self, x, *y):
        if x < 0:
            return x
        return 0

    def dequant_model(self):
        pass

    def quant(self, x, *y):
        if x >= self.edge_value[0]:
            return self.quant_value[0]
        if x <= self.edge_value[1]:
            return self.quant_value[1]*-1
        return 0

    def dequant(self, x, *y):
        pass

# 非对称正负稀疏三元量化 (a, 0 ,b) |a| != |b|
class STPandNQuanter:
    def __init__(self):
        self.QuantedModelStateDict: OrderedDict[str, torch.Tensor] = None
        self.DeQuantedModelStateDict: OrderedDict[str, torch.Tensor] = None

        self.layers_scale_zero_point = {}  # str: [quant_value, bool] true = pos, false = neg
        self.spare_rate = 0.05

        self.quant_value = [0, 0]
        self.symbol = None
        self.edge_value = [0, 0]

    def set_spare_rate(self, new_spare_rate):
        self.spare_rate = new_spare_rate

    def find_top_positive(self):
        pass

    def init_layers(self, model: nn.Module):
        for name, module in model.named_parameters():
            self.layers_scale_zero_point[name] = [0, None]

    def quant_layer(self, layer_param):
        mean_values, self.edge_value = self.get_layer_quant_value(copy.deepcopy(layer_param))
        # print(quant_value, _, self.edge_value)
        self.quant_value = mean_values

        new_param = layer_param.clone()
        quanted_param = new_param.map_(new_param, self.quant)
        return quanted_param

    def quant_model(self, model_state_dict):
        self.QuantedModelStateDict = copy.deepcopy(model_state_dict)
        for name, param in model_state_dict.items():

            if name == args.deep_model_layer_name:
                continue
            self.QuantedModelStateDict[name] = self.quant_layer(param)
        return self.QuantedModelStateDict

    #  非对称的 top_k
    def get_layer_quant_value(self, layer_tensor):
        # 计算每层的 稀疏量化值
        cat_tensor = layer_tensor.reshape(-1)
        top_size = math.ceil(cat_tensor.size()[0] * self.spare_rate)

        abs_cat_tensor = torch.abs(cat_tensor)

        abs_top_values = torch.topk(abs_cat_tensor, top_size, largest=True)

        top_tensor = torch.take(cat_tensor, abs_top_values[1])

        pos_list = []
        neg_list = []

        for v in top_tensor:
            if v > 0:
                pos_list.append(v)
            elif v < 0:
                neg_list.append(v)

        if len(pos_list) == 0:
            pos_mean = 0
            min_pos = 0
        else:
            pos_mean = np.mean(pos_list)
            min_pos = min(pos_list)


        if len(neg_list) == 0:
            neg_mean = 0
            max_neg = 0
        else:
            neg_mean = np.mean(neg_list)
            max_neg = max(neg_list)


        # print('------------------')
        # print(pos_top_values)
        #
        # print(neg_top_values)
        # print('------------------')


        return [pos_mean, neg_mean], [min_pos, max_neg]

    def map_pos(self, x, *y):
        if x > 0:
            return x
        return 0

    def map_neg(self, x, *y):
        if x < 0:
            return x
        return 0

    def dequant_model(self):
        pass

    def quant(self, x, *y):
        if x >= self.edge_value[0]:
            return self.quant_value[0]
        if x <= self.edge_value[1]:
            return self.quant_value[1]
        return 0

    def dequant(self, x, *y):
        pass

# 稀疏二元量化 (a, 0)
class SpareBinaryQuanter:
    def __init__(self):
        self.QuantedModelStateDict:OrderedDict[str, torch.Tensor] = None
        self.DeQuantedModelStateDict:OrderedDict[str, torch.Tensor] = None

        self.layers_scale_zero_point = {} # str: [quant_value, bool] true = pos, false = neg
        self.spare_rate = 0.05

        self.quant_value = 0
        self.symbol = None
        self.edge_value = 0

    def set_spare_rate(self, new_spare_rate):
        self.spare_rate = new_spare_rate

    def find_top_positive(self):

        pass

    def init_layers(self, model:nn.Module):
        for name, module in model.named_parameters():
            self.layers_scale_zero_point[name] = [0, None]


    def quant_layer(self, layer_param):
        quant_value, _, self.edge_value = self.get_layer_quant_value(copy.deepcopy(layer_param))
        # print(quant_value, _, self.edge_value)
        self.quant_value = quant_value
        self.symbol = _

        new_param = layer_param.clone()
        quanted_param = new_param.map_(new_param, self.quant)
        return quanted_param

    def quant_model(self, model_state_dict):
        self.QuantedModelStateDict = copy.deepcopy(model_state_dict)
        for name, param in model_state_dict.items():

            if name == args.deep_model_layer_name:
                continue
            quant_value, _, self.edge_value = self.get_layer_quant_value(copy.deepcopy(param))
            # print(quant_value, _, self.edge_value)

            self.layers_scale_zero_point[name] = [quant_value, _]
            self.quant_value = quant_value
            self.symbol = _

            new_param = param.clone()
            quanted_param = new_param.map_(new_param, self.quant)
            self.QuantedModelStateDict[name] = quanted_param
        return self.QuantedModelStateDict


    def get_layer_quant_value(self, layer_tensor):
        # 计算每层的 稀疏量化值
        cat_tensor = layer_tensor.reshape(-1)
        top_size = math.ceil(cat_tensor.size()[0] * self.spare_rate)

        pos_top_values = torch.topk(cat_tensor, top_size, largest=True)
        neg_top_values = torch.topk(cat_tensor, top_size, largest=False)
        # print('------------------')
        # print(pos_top_values)
        #
        # print(neg_top_values)
        # print('------------------')

        pos_mean = torch.mean(pos_top_values[0])
        neg_mean = torch.mean(neg_top_values[0])



        if torch.abs(pos_mean) > torch.abs(neg_mean):
            return pos_mean, True, torch.min(pos_top_values[0])
        else:
            return neg_mean, False, torch.max(neg_top_values[0])

    #  非对称的 top_k
    def get_layer_quant_value_PandN(self, layer_tensor):
        # 计算每层的 稀疏量化值
        cat_tensor = layer_tensor.reshape(-1)
        top_size = math.ceil(cat_tensor.size()[0] * self.spare_rate)

        abs_cat_tensor = torch.abs(cat_tensor)

        abs_top_values = torch.topk(abs_cat_tensor, top_size, largest=True)

        top_tensor = torch.take(cat_tensor, abs_top_values[1])

        pos_list = []
        neg_list = []

        for v in top_tensor:
            if v > 0:
                pos_list.append(v)
            elif v < 0:
                neg_list.append(v)

        pos_mean = np.mean(pos_list)
        neg_mean = np.mean(neg_list)


        # print('------------------')
        # print(pos_top_values)
        #
        # print(neg_top_values)
        # print('------------------')
        return pos_mean, neg_mean

    def map_pos(self, x, *y):
        if x > 0:
            return  x
        return 0

    def map_neg(self, x, *y):
        if x < 0:
            return  x
        return 0

    def dequant_model(self):
        pass


    def quant(self, x, *y):
        if self.symbol and x >= self.edge_value:
            return self.quant_value
        elif self.symbol is False and x <= self.edge_value:
            return self.quant_value
        else:
            return 0


    def spare(self, x, *y):
        if self.symbol and x >= self.edge_value:
            return x
        elif self.symbol is False and x <= self.edge_value:
            return x
        else:
            return 0


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
            quanted_param = new_param.map_(new_param, self.quant).to(torch.int8)
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
        self.Quanter = None

    def set_quanter(self, Quanter):
        self.Quanter = Quanter
        self.Quanter.init_layers(self)

    def quant(self):
        if self.Quanter is not None:
        # 量化模型权重 返回模型字典
            return self.Quanter.quant_model(self.state_dict())

    def dequant(self):
        if self.Quanter is not None:
            return self.Quanter.dequant_model(self.state_dict())

    def quant_test(self):
        if self.Quanter is not None:
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
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 10)


    def share_memory(self):
        for param in self.parameters():
            param.share_memory_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # x= self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # x = self.pool(x)
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

