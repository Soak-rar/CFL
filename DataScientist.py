import numpy as np
import torch
import Args
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas
import threading
import multiprocessing
import data.ff


def pca_dim_deduction(high_dim_data, max_dim):
    pca = PCA(n_components=max_dim)
    return pca.fit_transform(high_dim_data)


def pca_deduce(data_):
    # pca = PCA(n_components=dim)
    # new_data = pca.fit_transform(data_)
    _data = np.array(data_)
    # print(data_)
    # _data = [data_[i].mean(1) for i in range(len(data_))]
    return _data.mean(1)


def draw_3d_points(data_, point_style_list):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for id, data in enumerate(data_):
        ax.scatter(data[0], data[1], data[2], c=point_style_list[id % len(point_style_list)], marker="^")
    plt.title("10-Class Data Distribution")
    plt.show()


def draw_2d():
    args = Args.Arguments()
    global_1_1 = torch.load('0.4_0.4_mnist_cluster/Experiment_FedAvg0/Global.pt')
    global_1_2 = torch.load('0.4_0.4_mnist_cluster/Experiment1/Global.pt')

    global_2_1 = torch.load('0.4_0.4_cifar10_cluster/Experiment_FedAvg0/Global.pt')
    global_2_2 = torch.load('0.4_0.4_cifar10_cluster/Experiment0/Global.pt')
    fig = plt.figure(figsize=(24, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.arange(len(global_1_1['acc'])), global_1_1['acc'], label="random selection")
    ax1.plot(np.arange(len(global_1_2['acc'])), global_1_2['acc'], label="manual selection")
    # ax1.plot(np.arange(len(global_1_3['acc'])), global_1_3['acc'], label="10-Class Data Distribution")
    ax1.legend(loc='best')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Acc')
    ax1.set_title("mnist dataset")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.arange(len(global_2_1['acc'])), global_2_1['acc'], label="random selection")
    ax2.plot(np.arange(len(global_2_2['acc'])), global_2_2['acc'], label="manual selection")
    # ax2.plot(np.arange(len(global_2_3['acc'])), global_2_3['acc'], label="10-Class Data Distribution")
    ax2.legend(loc='best')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Acc')
    ax2.set_title("cifar10 dataset")
    plt.show()


# args = Args.Arguments()
# Global_1 = torch.load('0.4_0.4_mnist_cluster/实验1/Global.pt')
#
# Global_2 = torch.load('0.4_0.4_mnist_cluster/实验0/Global.pt')
#
# Global_3 = torch.load('0.4_0.4_cifar10_cluster/实验4/Global_Cost.pt')
#
# Global_4 = torch.load('0.4_0.4_cifar10_cluster/实验5/Global_Cost.pt')
#
#
# fig = plt.figure(figsize=(24, 6))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# # ax3 = fig.add_subplot(1, 3, 3)
#
#
# ax1.plot(np.arange(len(Global_1['acc'])), Global_1['acc'], label="Fed_Avg_10/100")
# ax1.plot(np.arange(len(Global_2['acc'])), Global_2['acc'], label="cluster-clients-2")
#
# ax1.legend(loc='best')
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('Acc')
# # plt.xticks(range(0, 50))
#
# ax2.plot(np.arange(len(Global_3)), Global_3, label="cifar10-clients_10/100")
# ax2.plot(np.arange(len(Global_4)), Global_4, label="cluster-clients-2")
# #
# ax2.legend(loc='best')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('Cost')
#
# # for i, cost in enumerate(results[:, 2]):
# #     ax3.plot(np.arange(len(cost)), [cost[0]*(i+1) for i in range(len(cost))], label=dirlist[i])
# #
# # ax3.legend(loc='best')
# # ax3.set_xlabel('epoch')
# # ax3.set_ylabel('Cost  M')
# # plt.xticks(range(0, 50))
# plt.show()
if __name__ == '__main__':
    style_list = ["b", "g"]
    data_list = np.array([[1,2,3],[4, 1, 2],[3,5, 4],[5,6,7], [7,8,9],[0, 9,1]])
    new_data_list = [[1], [2], [3], [4], [1]]

    print(pca_dim_deduction(new_data_list, 2))
    # draw_2d()


