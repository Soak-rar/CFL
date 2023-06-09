import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import random
import numpy as np
import threading
import matplotlib.pyplot as plt
import os
import Args


def generate(args, dataset_name='mnist'):
    data_target, data_target_test = Data_Process(args, dataset_name)

    local_classes = int(args.local_data_classes * args.dataset_labels_number)
    class_index = 0
    for i in range(args.cluster_number):
        local_class_index_list = [(j + class_index) % 10 for j in range(local_classes)]
        local_class_index_list.sort()

        cluster_data_test = []
        cluster_label_test = []

        for label in local_class_index_list:
            cluster_data_test.extend(data_target_test[label])
            cluster_label_test.extend(np.full(len(data_target_test[label]), label))

        torch.save({'data': torch.Tensor(cluster_data_test),
                    'label': torch.Tensor(cluster_label_test),
                    'data_len': len(cluster_data_test)},
                   args.save_path + '/cluster_test_' + str(i) + '.pt')

        class_index = (class_index + local_classes) % 10

    class_index = 0

    for i in range(args.worker_num):

        # local_class_index_list = random.sample(range(classes_num), local_classes)
        local_class_index_list = [(j+class_index) % 10 for j in range(local_classes)]
        local_class_index_list.sort()
        print(local_class_index_list)
        class_index = (class_index + local_classes) % 10

        local_data = []
        local_labels = []
        print('生成客户端 ' + str(i) + '数据集...')
        data_len = {key: 0 for key in range(args.dataset_labels_number)}

        for index in local_class_index_list:
            local_size = int(args.local_data_size * len(data_target[index]))


            local_data_with_label = random.sample(data_target[index], local_size)

            local_data.extend(local_data_with_label[:])
            local_labels.extend(np.full(local_size, index))

            data_len[index] = local_size


        torch.save({'data': torch.Tensor(local_data),
                    'label': torch.Tensor(local_labels),
                    'data_len': data_len},
                   args.save_path + '/train_worker_' + str(i) + '.pt')


def load_data(args: Args.Arguments):
    train_workers = []
    test_clusters = []
    for i in range(args.worker_num):
        train_workers.append(torch.load('../' + args.save_path + '/train_worker_' + str(i) + '.pt'))
    for i in range(args.cluster_number):
        test_clusters.append(torch.load('../' + args.save_path + '/cluster_test_' + str(i) + '.pt'))
    return train_workers, test_clusters


# 使用Args中的 data_Dis生成数据
def generate_data_with_data_self(args: Args.Arguments):
    data = Data_Process(args, args.dataset_name)

    if data is not None:
        data_target, data_target_test = data
    else:
        return None

    local_classes = int(args.local_data_classes * args.dataset_labels_number)
    class_index = 0
    for i in range(args.cluster_number):
        local_class_index_list = args.data_Dis[class_index]['data_labels']

        cluster_data_test = []
        cluster_label_test = []

        for label in local_class_index_list:
            cluster_data_test.extend(data_target_test[label])
            cluster_label_test.extend(np.full(len(data_target_test[label]), label))

        torch.save({'data': torch.Tensor(cluster_data_test),
                    'label': torch.Tensor(cluster_label_test),
                    'data_len': len(cluster_data_test)},
                   args.save_path + '/cluster_test_' + str(i) + '.pt')

        class_index += 1
        class_index %= len(args.data_Dis)

    class_index = 0
    for i in range(args.worker_num):
        # local_class_index_list = random.sample(range(classes_num), local_classes)
        local_class_index_list = args.data_Dis[class_index]['data_labels']
        print(local_class_index_list)
        class_index += 1
        class_index %= len(args.data_Dis)

        local_data = []
        local_labels = []
        print('生成客户端 ' + str(i) + '数据集...')
        data_len = {key: 0 for key in range(args.dataset_labels_number)}

        for index in local_class_index_list:
            local_size = int(args.data_Dis[class_index]['label_len'] * len(data_target[index]))

            local_data_with_label = random.sample(data_target[index], local_size)

            local_data.extend(local_data_with_label[:])
            local_labels.extend(np.full(local_size, index))

            data_len[index] = local_size

        torch.save({'data': torch.Tensor(local_data),
                    'label': torch.Tensor(local_labels),
                    'data_len': data_len},
                   args.save_path + '/train_worker_' + str(i) + '.pt')


def generate_rotated_data(mArgs: Args.Arguments):
    DataSet = {}
    dataset = {}
    dataset['data_indices'], dataset['cluster_assign'] = \
        _setup_dataset(60000, mArgs.cluster_number, mArgs.worker_num, mArgs.rot_local_data_size)
    print(dataset['cluster_assign'])
    (X, y) = _load_MNIST(train=True)
    dataset['data_rot'] = dataset['cluster_assign'][:]
    dataset['X'] = X
    dataset['y'] = y
    DataSet['train'] = dataset
    # print('gen', X)

    dataset = {}
    dataset['data_indices'], dataset['cluster_assign'] = \
        _setup_dataset(10000, mArgs.cluster_number, mArgs.test_worker_num, mArgs.rot_local_data_size, random=False)
    (X, y) = _load_MNIST(train=False)

    dataset['data_rot'] = dataset['cluster_assign'][:]
    dataset['X'] = X
    dataset['y'] = y
    DataSet['test'] = dataset

    return DataSet

def _load_MNIST(train=True):
    Transforms = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                              (0.1307,), (0.3081,))
                         ])
    if train:
        mnist_dataset = datasets.MNIST(root='data', train=True, download=True)
    else:
        mnist_dataset = datasets.MNIST(root='data', train=False, download=True)

    # data_set = []
    # label_set = []
    # for i, data in enumerate(mnist_dataset.data):
    #     data_set.append(Transforms(data.numpy()).numpy())
    #     label_set.append(mnist_dataset.targets[i])
    #
    # X = torch.tensor(data_set) # (60000,28, 28)
    # y = torch.tensor(label_set) #(60000)

    dl = DataLoader(mnist_dataset)

    X = dl.dataset.data  # (60000,28, 28)
    y = dl.dataset.targets  # (60000)

    # normalize to have 0 ~ 1 range in each pixel

    X = X / 255.0

    return X, y

def _setup_dataset(num_data, p, m, n, random = True):
    assert (m // p) * n == num_data

    data_indices = []
    cluster_assign = []

    m_per_cluster = m // p

    for p_i in range(p):

        if random:
            ll = list(np.random.permutation(num_data))
        else:
            ll = list(range(num_data))

        ll2 = chunkify(ll, m_per_cluster) # splits ll into m lists with size n
        data_indices += ll2

        cluster_assign += [p_i for _ in range(m_per_cluster)]

    data_indices = np.array(data_indices)
    cluster_assign = np.array(cluster_assign)

    assert data_indices.shape[0] == cluster_assign.shape[0]
    assert data_indices.shape[0] == m

    return data_indices, cluster_assign



def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(gen)

def Data_Process(args, dataset_name):
    dataset_len = 0
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    else:
        # ex = Exception('文件夹'+args.save_path+'已经存在')
        print('文件夹' + args.save_path + '已经存在')
        return None

    # 初始化数据工程变量
    if dataset_name == 'mnist':
        # 读取数据集
        dataset = datasets.MNIST('data', train=True, download=True)
        dataset_test = datasets.MNIST('data', train=False, download=True)
        # 数据集转换标准化
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_len = len(dataset)
        classes_num = len(dataset.classes)
        # 将数据集按类别重新整理，并做标准化操作
        data_target = [[] for key in range(classes_num)]

        for (data, target) in zip(dataset.data, dataset.targets):
            data_target[target.item()].append(trans(data.numpy()).numpy())

        data_target_test = [[] for key in range(classes_num)]
        for (data, target) in zip(dataset_test.data, dataset_test.targets):
            data_target_test[target.item()].append(trans(data.numpy()).numpy())


    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10('data', train=True, download=True)
        dataset_test = datasets.CIFAR10('data', train=False, download=True)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        classes_num = len(dataset.classes)
        # 将数据集按类别重新整理，并做标准化操作
        data_target = [[] for key in range(classes_num)]

        for (data, target) in zip(dataset.data, dataset.targets):
            data_target[target].append(trans(data).numpy())

        data_target_test = [[] for key in range(classes_num)]
        for (data, target) in zip(dataset_test.data, dataset_test.targets):
            data_target_test[target].append(trans(data).numpy())
    else:
        return None

    return data_target, data_target_test, dataset_len


if __name__ == '__main__':
    args = Args.Arguments()
    generate_rotated_data(args)
    # if len(args.data_Dis) > 0:
    #     generate_data_with_data_self(args)
    # else:
    #     generate(args, args.dataset_name)

