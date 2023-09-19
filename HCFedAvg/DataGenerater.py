# 读取并 预处理数据集， 同时根据需求生成 客户端 数据索引列表
import copy
from typing import *

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import random
import numpy as np
import Args
from torchvision.utils import save_image


class ClientDataInfo:
    def __init__(self, in_cluster_id, client_id, is_rot = False, rot = 0):
        self.InClusterID = in_cluster_id
        self.ClientID = client_id
        self.DataIndex = []
        self.DatasetLoader = None

        # 是否旋转， 旋转 rot * 90度
        self.IsRot = is_rot
        self.Rot = rot


class DatasetGen:
    def __init__(self, args: Args.Arguments):
        # 预处理后的数据 列表
        self.Data = None
        # 预处理后的数据 标签列表
        self.Label = None

        self.LabelDataIndex = None
        self.TestLabelDataIndex = None

        # 预处理后的 测试集 数据 和 标签
        self.TestData = None
        self.TestLabel = None

        self.Trans = None

        # 客户端的 本地数据标签
        self.ClientsDataInfo: List[ClientDataInfo] = []
        self.mArgs: Args.Arguments = args

        self.ClusterTestDataIndex = [[] for i in range(self.mArgs.cluster_number)]

        self.init_client_data_list()
        self.normalize_dataset()
        self.divide_clients_data_index()

    def print_img(self):
        for i in range(4):
            rot_imgs = torch.rot90(self.Data[0][0], k=int(i))
            save_image(rot_imgs, "%d.png" % i, nrow=5, normalize=True)


    # 将全部集群的测试集合并为 FedAvg的测试集数据
    def get_fedavg_test_DataLoader(self):
        data_index = []
        for index in self.ClusterTestDataIndex:
            data_index.extend(index)
        data = self.TestData[data_index]
        label = self.TestLabel[data_index]
        test_dataset = TensorDataset(self.TestData,
                                     self.TestLabel)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
        return test_loader

    def get_client_DataLoader(self, client_id):
        ClientInfo = self.ClientsDataInfo[client_id]
        data_index = ClientInfo.DataIndex
        data = self.Data[data_index]
        label = self.Label[data_index]


        # 如果是 'rot' 旋转数据生成Loader
        if ClientInfo.IsRot:
            if ClientInfo.Rot > 0:

                X_batch2 = torch.rot90(data, k=int(ClientInfo.Rot), dims=(2, 3))
                train_dataset = TensorDataset(X_batch2,
                                              label)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
                return train_loader
            else:
                train_dataset = TensorDataset(data,
                                              label)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
                return train_loader
        # 如果是 'labels' 直接生成Loader
        else:
            train_dataset = TensorDataset(data,
                                          label)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
            return train_loader

    def get_cluster_test_DataLoader(self, cluster_id):
        if self.mArgs.data_info["divide_type"] == 'labels':

            cluster_labels = self.mArgs.data_info['data_labels'][cluster_id]
            data_index = []
            for label in cluster_labels:
                data_index.extend(self.TestLabelDataIndex[label])

            test_data = self.TestData[data_index]
            test_label = self.TestLabel[data_index]
            test_dataset = TensorDataset(test_data,
                                         test_label)

            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
            return test_loader
        elif self.mArgs.data_info["divide_type"] == 'rot':
            rot = self.mArgs.data_info['data_rot'][cluster_id]
            data = copy.deepcopy(self.TestData)
            label = copy.deepcopy(self.TestLabel)
            if rot > 0:
                X_batch2 = torch.rot90(data, k=int(rot), dims=(2, 3))
                test_dataset = TensorDataset(X_batch2,
                                              label)
            else:
                test_dataset = TensorDataset(data,
                                             label)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
            return test_loader

        # data_index = self.ClusterTestDataIndex[cluster_id]
        #
        #
        # data = self.TestData[data_index]
        # label = self.TestLabel[data_index]
        # test_dataset = TensorDataset(data,
        #                               label)
        #
        # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
        # return test_loader

    def init_client_data_list(self):
        cluster_num = self.mArgs.cluster_number
        # assert cluster_num == len(self.mArgs.data_info['data_labels'])
        # assert cluster_num == len(self.mArgs.data_info['data_rot'])

        for i in range(self.mArgs.worker_num):
            if self.mArgs.data_info['divide_type'] == 'rot':
                self.ClientsDataInfo.append(ClientDataInfo((i % cluster_num), i, True, self.mArgs.data_info['data_rot'][i % cluster_num]))
            elif self.mArgs.data_info['divide_type'] == 'labels':
                self.ClientsDataInfo.append(ClientDataInfo((i % cluster_num), i))
    # 加载预处理数据集
    def load_dataset(self):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if self.mArgs.dataset_name == 'mnist':
            dataset = datasets.MNIST('data', train=True, download=True, transform=trans)
            dataset_test = datasets.MNIST('data', train=False, download=True, transform=trans)
            # 数据集转换标准化

        elif self.mArgs.dataset_name == 'cifar10':
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.CIFAR10('data', train=True, download=True, transform=trans)
            dataset_test = datasets.CIFAR10('data', train=False, download=True, transform=trans)

        else:
            dataset = None
            dataset_test = None

        if dataset is not None:
            data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
            test_data_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)
        else:
            data_loader = None
            test_data_loader = None

        return data_loader, test_data_loader


    def normalize_dataset(self):
        data_loader, test_data_loader = self.load_dataset()
        assert data_loader is not None

        data_list = []
        label_list = []

        test_data_list = []
        test_label_list = []

        for data_, label_ in iter(data_loader):
            data_list.append(data_[0].numpy())
            label_list.append(label_[0].numpy())

        random.seed(15)
        random.shuffle(data_list)

        random.seed(15)
        random.shuffle(label_list)

        for data_, label_ in iter(test_data_loader):
            test_data_list.append(data_[0].numpy())
            test_label_list.append(label_[0].numpy())

        self.Data = torch.tensor(np.array(data_list))
        self.Label = torch.tensor(np.array(label_list))
        self.TestData = torch.tensor(np.array(test_data_list))
        self.TestLabel = torch.tensor(np.array(test_label_list))

        self.LabelDataIndex = [[] for _ in range(self.mArgs.dataset_labels_number)]
        for i, label in enumerate(label_list):
            self.LabelDataIndex[label.item()].append(i)

        self.TestLabelDataIndex = [[] for _ in range(self.mArgs.dataset_labels_number)]
        for i, label in enumerate(test_label_list):
            self.TestLabelDataIndex[label.item()].append(i)


    def divide_clients_data_index(self):
        if self.mArgs.data_info['divide_type'] == 'labels':
            cluster_labels = self.mArgs.data_info['data_labels']
            MaxLen = 0
            for labels_list in cluster_labels:
                if len(labels_list) > MaxLen:
                    MaxLen = len(labels_list)

            # 计算每个集群的 每种标签的数据量占比
            labels_ratio = [[] for i in range(self.mArgs.dataset_labels_number)]
            for labels_list in cluster_labels:
                for label in labels_list:
                    labels_ratio[label].append(1.0 / len(labels_list))

            MaxRatio = 0
            MaxLabel = 0
            MinDataSize = 99999
            for label, label_ratio in enumerate(labels_ratio):
                if sum(label_ratio) > MaxRatio or (
                        sum(label_ratio) == MaxRatio and MinDataSize > len(self.TestLabelDataIndex[label])):
                    MinDataSize = len(self.TestLabelDataIndex[label])
                    MaxRatio = sum(label_ratio)
                    MaxLabel = label

            ClusterDataSize = int(len(self.TestLabelDataIndex[MaxLabel]) / MaxRatio)

            labels_current_pos = [0 for i in range(self.mArgs.dataset_labels_number)]
            test_cluster_labels_list = {i: {j: [] for j in cluster_labels[i]} for i in range(self.mArgs.cluster_number)}
            for cluster_id, cluster_label in test_cluster_labels_list.items():
                for label_id, label_index in cluster_label.items():
                    labels_len = len(cluster_labels[cluster_id])
                    start_pos = labels_current_pos[label_id]
                    end_pos = int(ClusterDataSize / labels_len) + start_pos
                    test_cluster_labels_list[cluster_id][label_id] = self.TestLabelDataIndex[label_id][start_pos: end_pos]
                    self.ClusterTestDataIndex[cluster_id].extend(self.TestLabelDataIndex[label_id][start_pos: end_pos])
                    labels_current_pos[label_id] = end_pos

            for cluster_id, cluster_label in test_cluster_labels_list.items():
                for label_id, label_index in cluster_label.items():
                    random.shuffle(label_index)

            MaxRatio = 0
            MaxLabel = 0
            MinDataSize = 99999
            for label, label_ratio in enumerate(labels_ratio):
                if sum(label_ratio) > MaxRatio or (sum(label_ratio) == MaxRatio and MinDataSize > len(self.LabelDataIndex[label])):
                    MinDataSize = len(self.LabelDataIndex[label])
                    MaxRatio = sum(label_ratio)
                    MaxLabel = label

            ClusterDataSize = int(len(self.LabelDataIndex[MaxLabel]) / MaxRatio)

            labels_current_pos = [0 for i in range(self.mArgs.dataset_labels_number)]
            cluster_labels_list = {i: {j: [] for j in cluster_labels[i]} for i in range(self.mArgs.cluster_number)}
            for cluster_id, cluster_label in cluster_labels_list.items():
                for label_id, label_index in cluster_label.items():
                    labels_len = len(cluster_labels[cluster_id])
                    start_pos = labels_current_pos[label_id]
                    end_pos = int(ClusterDataSize / labels_len) + start_pos
                    cluster_labels_list[cluster_id][label_id] = self.LabelDataIndex[label_id][start_pos: end_pos]
                    labels_current_pos[label_id] = end_pos

            cluster_clients_list = {i: [] for i in range(self.mArgs.cluster_number)} # 每个集群 客户端数量
            for client_data in self.ClientsDataInfo:
                cluster_clients_list[client_data.InClusterID].append(client_data.ClientID)

            for cluster_id in range(self.mArgs.cluster_number):
                cluster_client_num = len(cluster_clients_list[cluster_id])
                for label, labels_list in cluster_labels_list[cluster_id].items():
                    label_size = len(labels_list)//cluster_client_num
                    start_pos = 0
                    end_pos = label_size
                    for client_id in cluster_clients_list[cluster_id]:
                        self.ClientsDataInfo[client_id].DataIndex.extend(labels_list[start_pos: end_pos])
                        start_pos = end_pos
                        end_pos += label_size

        elif self.mArgs.data_info['divide_type'] == 'rot':
            cluster_number = self.mArgs.cluster_number
            assert cluster_number == len(self.mArgs.data_info['data_rot'])
            client_number_pre_cluster = self.mArgs.worker_num // cluster_number

            label_pre_len = {}

            for label, label_list in enumerate(self.LabelDataIndex):
                label_pre_len[label] = len(label_list) // client_number_pre_cluster


            clusters_start_pos = [[0 for _ in range(self.mArgs.worker_num) ] for i in range(cluster_number)]

            for client_data in self.ClientsDataInfo:
                for label, label_list in enumerate(self.LabelDataIndex):
                    start_pos = clusters_start_pos[client_data.InClusterID][label]
                    end_pos = start_pos + label_pre_len[label]
                    client_data.DataIndex.extend(label_list[start_pos: end_pos])
                    clusters_start_pos[client_data.InClusterID][label] = end_pos

        else:
            pass



if __name__ == '__main__':
    args = Args.Arguments()
    DatasetGen(args)







