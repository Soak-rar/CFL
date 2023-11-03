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
from HCFedAvg import FileProcess
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
        return math.pow(Value, 0.5)

    else:
        relative_tensor = (tensor1 - tensor2)/tensor1
        Value = 0
        for i in range(relative_tensor.shape[0]):
            Value += math.pow(relative_tensor[i].item(), 2)
        return Value



def avg_deep_param(model_dict, init_model_dict,pre_model_dict=False):
    AvgParam = torch.zeros(model_dict.shape[0])
    for i in range(model_dict.shape[1]):
        for j in range(model_dict.shape[0]):
            if pre_model_dict:
                AvgParam[j] = AvgParam[j] + (model_dict[j][i])
            else:
                AvgParam[j] = AvgParam[j] + (model_dict[j][i] - init_model_dict[j][i])
            # AvgParam[j] = AvgParam[j] + (model_dict[j][i])
    return AvgParam / model_dict.shape[1]
    # return model_dict[k][:].cpu()


def avg_deep_param_bias(model_dict, init_model_dict, args, pre_model_dict=None):

    return model_dict - init_model_dict
    # return model_dict[k][:].cpu()


def tensor_normal(tensor_):


    norm = torch.norm(tensor_, p=2)

    # 进行向量单位化
    normalized_x = tensor_ / norm

    return normalized_x


if __name__ == '__main__':
    init_model = torch.load('test_model/init_model_dict.pth')
    avg_models = torch.load('test_model/avg_model_deep_.pth')
    signal_models_95 = torch.load('test_model/signal_model_deep_95.pt')
    signal_models_90 = torch.load('test_model/signal_model_deep_90.pt')
    args = Args.Arguments()

    header = {"data": "", "data_type": "", "TAS": [], "COS": []}
    name = "TAS_result"
    # final_deep = tensor_normal(final_deep)
    header['data'] = args.dataset_name
    header['data_type'] = args.data_info["data_labels"]
    # final_model = avg_models[-1]
    final_model = signal_models_90[-1]
    final_deep_1 = avg_deep_param(final_model[args.deep_model_layer_name], init_model[args.deep_model_layer_name])
    final_deep_2 = avg_deep_param(avg_models[95][args.deep_model_layer_name], init_model[args.deep_model_layer_name])
    final_deep_3 = avg_deep_param(final_model[args.deep_model_layer_name], avg_models[95][args.deep_model_layer_name], True)

    final_deep_4 = avg_deep_param(final_model[args.deep_model_layer_name], init_model[args.deep_model_layer_name], True)

    dis_list_1 = []
    dis_list_2 = []
    dis_list_3 = []
    for index, signal_model in enumerate(signal_models_95):
        deep_1 = avg_deep_param(signal_model[args.deep_model_layer_name], init_model[args.deep_model_layer_name])
        deep_2 = avg_deep_param(avg_models[index * 5][args.deep_model_layer_name], init_model[args.deep_model_layer_name])

        # deep_ = tensor_normal(deep_)
        dis_value_1 = L2_Distance(final_deep_1, deep_1)
        dis_value_2 = L2_Distance(final_deep_2, deep_2)
        dis_list_1.append(abs(dis_value_1-dis_value_2))

        deep_3 = avg_deep_param(signal_model[args.deep_model_layer_name], avg_models[index * 5][args.deep_model_layer_name], True)
        dis_value_3 = L2_Distance(final_deep_4, deep_3)
        dis_list_2.append(dis_value_3)

        dis_value_4 = L2_Distance(final_deep_4, deep_3, Use_cos=1)
        dis_list_3.append(dis_value_4)

    # FileProcess.read_row_with_file_name(2,'TAS_result')
    fig = plt.figure(figsize=(24, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    header["TAS"] = dis_list_1[::-1]
    header["COS"] = dis_list_2[::-1]
    header["L2"] = dis_list_3[::-1]

    # FileProcess.add_row_with_file_name(header, "HCFedAvg/TAS_result")
    plt.rcParams.update({'font.size': 18})
    ax1.plot(range(len(dis_list_1)), dis_list_1[::-1], label="TAS")
    ax1.plot(range(len(dis_list_2)), dis_list_2[::-1], label="COS")
    ax1.plot(range(len(dis_list_3)), dis_list_3[::-1], label="L2")

    ax1.legend(loc= 'upper left', prop={'size': 18})
    plt.xlabel('Time Difference', fontsize=16)
    plt.ylabel('Similarity Distance', fontsize=16)
    x_ticks = [i for i in range(21)]
    x_labels = [str(i) for i in range(21)]
    plt.xticks(x_ticks, x_labels, fontsize=16)
    plt.title('Distribution-Parallel', fontsize=24)
    plt.show()


