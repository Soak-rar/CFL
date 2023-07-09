import numpy as np
import torch
import Model
from torchvision import datasets, transforms
import Args
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas
import threading
import multiprocessing
import data.ff
from HCFedAvg.DataGenerater import *
from torchsummary import summary
import math


def L2_Distance(tensor1, tensor2, Use_cos = 0):

    if Use_cos == 0:
        UpSum = 0
        for i in range(tensor1.shape[0]):
            UpSum += tensor1[i].item() * tensor2[i].item()
        DownSum1 = 0
        DownSum2 = 0
        for i in range(tensor1.shape[0]):
            DownSum1 += tensor1[i].item() * tensor1[i].item()
        DownSum1 = DownSum1 ** 0.5
        for i in range(tensor2.shape[0]):
            DownSum2 += tensor2[i].item() * tensor2[i].item()
        DownSum2 = DownSum2 ** 0.5

        return abs(1 - UpSum / (DownSum1 * DownSum2))
    elif Use_cos == 1:
        Value = 0
        for i in range(tensor1.shape[0]):
            Value += math.pow(tensor1[i].item() - tensor2[i].item(), 2)
        return Value

    else:
        relative_tensor = (tensor1 - tensor2)/tensor1
        Value = 0
        for i in range(relative_tensor.shape[0]):
            Value += math.pow(relative_tensor[i].item(), 2)
        return Value



def avg_deep_param(model_dict, init_model_dict, args, pre_model_dict=None):
    AvgParam = torch.zeros(model_dict.shape[0])
    for i in range(model_dict.shape[1]):
        for j in range(model_dict.shape[0]):
            if pre_model_dict is not None:
                AvgParam[j] = AvgParam[j] + (model_dict[j][i] - pre_model_dict[j][i])
            else:
                AvgParam[j] = AvgParam[j] + (model_dict[j][i] - init_model_dict[args.deep_model_layer_name][j][i])
            # AvgParam[j] = AvgParam[j] + (model_dict[j][i])
    return AvgParam / model_dict.shape[1]
    # return model_dict[k][:].cpu()


def tensor_normal(tensor_):


    norm = torch.norm(tensor_, p=2)

    # 进行向量单位化
    normalized_x = tensor_ / norm

    return normalized_x


if __name__ == '__main__':
    init_model = torch.load('HCFedAvg/test_model/init_model_dict.pth')
    avg_models = torch.load('HCFedAvg/test_model/avg_model_deep_.pth')
    signal_models_95 = torch.load('HCFedAvg/test_model/signal_model_deep_95.pt')
    signal_models_90 = torch.load('HCFedAvg/test_model/signal_model_deep_90.pt')
    args = Args.Arguments()



    # final_deep = tensor_normal(final_deep)

    # final_model = avg_models[-1]
    final_model = signal_models_90[-1]
    final_deep = avg_deep_param(final_model, init_model, args)
    dis_list = []
    for index, signal_model in enumerate(signal_models_95):
        deep_ = avg_deep_param(signal_model, init_model, args)

        # deep_ = tensor_normal(deep_)
        dis_value = L2_Distance(final_deep, deep_)
        dis_list.append(dis_value)
    fig = plt.figure(figsize=(24, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(range(len(dis_list)), dis_list, label="FedAvg")
    print(dis_list)
    plt.show()

