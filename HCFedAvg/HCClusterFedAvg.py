import collections
from typing import Dict
from DataGenerater import *
import gc
import HCClusterTree
import torch
import random
import copy
import math
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import Args
from Args import ClientInServerData
import KMeansPP
import datetime
import MMDLoss
import Model
import Data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time
import os
import torch
import Model
import torch.multiprocessing as mp
import DataGenerater
from torch.multiprocessing import SimpleQueue, Manager
import time
if mp.get_start_method() != 'spawn':
    mp.set_start_method('spawn')
HC_queue = SimpleQueue()
FedAvg_queue = SimpleQueue()
from DataScientist import pca_deduce


def train(model: torch.nn.Module, datasetLoader, worker_id, device, args, q: SimpleQueue, shared_models):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    model.to(device=device)
    data_size = 0
    avg_loss = 0
    count = 0
    loss_avg = 0
    for local_epoch in range(args.local_epochs):
        for batch_index, (batch_data, batch_label) in enumerate(datasetLoader):
            optimizer.zero_grad()

            if local_epoch == 0:
                data_size += len(batch_data)
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_data)
            # print(pred)
            loss = F.nll_loss(pred, batch_label)

            avg_loss += loss.item()
            count +=1

            loss.backward()

            optimizer.step()
    # print(worker_id)
    # print(avg_loss)
    model.to('cpu')
    # print(avg_loss/count)
    data_len = data_size
    cost = 0
    cost += sum([param.nelement() for param in model.parameters()])

    # loss, acc = test(model, test_loader, device)
    q.put({'data_len': data_len,
           'id': worker_id,
           'cost': cost * 4})
    shared_models[worker_id]['model'].load_state_dict(copy.deepcopy(model.state_dict()))

def test(model, dataset_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    model = model.to(device)
    with torch.no_grad():
        for data, target in dataset_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_len = len(dataset_loader.dataset)
    test_loss /= test_len
    model.to('cpu')
    return test_loss, correct / test_len


def avg(model_dict, local_model_dicts):
    total_len = 0
    for model_inf in local_model_dicts:
        total_len += model_inf['data_len']
    for key in model_dict.keys():
        model_dict[key] *= 0
        for remote_model in local_model_dicts:
            model_dict[key] += (remote_model['model_dict'][key] * remote_model['data_len'] / total_len)
    return model_dict


def create_process(model_1, model_2, DataGen:DatasetGen, worker_id, device, args, queue_1, queue_2, shared_models_1, shared_models_2):
    train(model_1, DataGen.get_client_DataLoader(worker_id), worker_id, device, args, queue_1, shared_models_1)
    train(model_2, DataGen.get_client_DataLoader(worker_id), worker_id, device, args, queue_2, shared_models_2)




def main(args):
    seed_ = 100
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    datasetGen = DatasetGen(args)
    # DataSet = Data.generate_rotated_data(args)

    manager = Manager()

    train_workers = [i for i in range(args.worker_num)]

    global_model_test_data = init_test_dataset_loader(args.dataset_name, args.batch_size)

    # FedAvg算法的模型
    FedAvg_global_model = Model.init_model(args.model_name)

    # 每个客户端的 局部模型，初始化时为相同的模型

    global_model = Model.init_model(args.model_name)

    avg_model = copy.deepcopy(global_model)

    FedAvg_global_model.load_state_dict(global_model.state_dict())

    clients_model: Dict[int, ClientInServerData] = {}

    pre_clients_model = {}

    ClusterManager = HCClusterTree.HCClusterManager()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    TotalLoss = []
    TotalAcc = []

    FedAvg_Loss = []
    FedAvg_Acc = []

    ProcessList = []

    old_matrix = {}
    for epoch in range(args.global_round):
        # print("Epoch :{}\t,".format(epoch + 1), end='')
        cluster_clients_train = random.sample(train_workers, args.worker_train)

        for c in cluster_clients_train:
            if c in train_workers:
                train_workers.remove(c)

        if len(train_workers) == 0:
            train_workers = [i for i in range(args.worker_num)]

        epoch_loss = []
        epoch_acc = []

        FedAvg_local_models = []
        print(' 参与训练的客户端 ：')
        print(cluster_clients_train)

        current_train_client_model_dict = manager.dict()
        fedavg_client_model = manager.dict()

        aliving_process_number = 0

        for i, worker_id in enumerate(cluster_clients_train):

            # print(epoch, "  worker : ", worker_id)

            FirstSelected = False

            if clients_model.get(worker_id) is None:
                clients_model[worker_id] = ClientInServerData(worker_id, global_model.state_dict(), worker_id, 1)
                train_model_dict = global_model.state_dict()
                FirstSelected = True

            else:
                current_cluster = ClusterManager.get_cluster_by_id(clients_model[worker_id].InClusterID)
                train_model_dict = current_cluster.get_avg_cluster_model_copy()
                # train_model_dict = clients_model[worker_id].ModelStaticDict
                clients_model[worker_id].TrainRound = current_cluster.CurrentModelRound + 1

            client_model = Model.init_model(args.model_name)
            fedavg_model = Model.init_model(args.model_name)
            client_model.load_state_dict(train_model_dict)
            fedavg_model.load_state_dict(FedAvg_global_model.state_dict())
            clients_model[worker_id].PreModelStaticDict = copy.deepcopy(train_model_dict)
            current_train_client_model_dict[worker_id] = {'model': client_model, 'first': FirstSelected}
            fedavg_client_model[worker_id] = {'model': fedavg_model, 'first': FirstSelected}

            local_p = mp.Process(target=create_process, args=(client_model, fedavg_model, datasetGen, worker_id, device, args, HC_queue, FedAvg_queue, current_train_client_model_dict, fedavg_client_model))
            ProcessList.append(local_p)
            if len(ProcessList) == args.MaxProcessNumber or i == len(cluster_clients_train)-1:
                for p in ProcessList:
                    p.start()
                for p in ProcessList:
                    p.join()
                ProcessList.clear()

            # print( loss,"   " ,acc)
            # epoch_loss.append(loss)
            # epoch_acc.append(acc)



        local_train_info = {}
        fedavg_local_train_info = {}
        while not HC_queue.empty():
            train_info = HC_queue.get()
            local_train_info[train_info['id']] = train_info
        while not FedAvg_queue.empty():
            train_info = FedAvg_queue.get()
            fedavg_local_train_info[train_info['id']] = train_info

        for worker_id, model in fedavg_client_model.items():
            train_eval = {'model_dict': model['model'].state_dict(), 'data_len': fedavg_local_train_info[worker_id]['data_len']}
            FedAvg_local_models.append(train_eval)

        for worker_id, model_dict in current_train_client_model_dict.items():
            clients_model[worker_id].set_client_info(model_dict['model'].state_dict(), local_train_info[worker_id]['data_len'])
            if model_dict['first']:
                clients_model[worker_id].PreModelStaticDict = copy.deepcopy(global_model.state_dict())
        fedavg_client_model.clear()
        current_train_client_model_dict.clear()
        # 计算相似性矩阵
        # similarity_matrix = update_client_similarity_matrix(clients_model, args)

        global_si_ma = calculate_relative_similarity(clients_model, global_model, cluster_clients_train, old_matrix, args)
        old_matrix = copy.deepcopy(global_si_ma)
        ClusterManager.reset_similarity_matrix(global_si_ma)
        std_m = []
        td_sm = copy.deepcopy(global_si_ma)
        for key, value in td_sm.items():
            value.pop(key)
            std_m.extend(value.values())
        std = np.mean(std_m)
        print('mean: ', std)

        ClusterManager.HCClusterDivide()

        # ClusterManager.print_divide_result()

        # ClusterManager.UpdateClusterAvgModel(clients_model, cluster_clients_train)
        ClusterManager.UpdateClusterAvgModelWithTime(clients_model, cluster_clients_train)
        epoch_loss = []
        epoch_acc = []

        ## 集群准确性测试
        for cluster_id, Cluster in ClusterManager.CurrentClusters.items():
            test_dataloader = datasetGen.get_cluster_test_DataLoader(cluster_id % args.cluster_number)
            test_model = Model.init_model(args.model_name)
            test_model.load_state_dict(Cluster.AvgClusterModelDict)
            loss, acc = test(test_model, test_dataloader, device)
            epoch_loss.append(loss)
            epoch_acc.append(acc)

        # 输出当前轮次集群结果
        trained = cluster_clients_train[:]
        print(' 轮次划分结果 ')
        for cluster_id, Cluster in ClusterManager.CurrentClusters.items():
            print('cluster_id: ', cluster_id, ' , res: ', end='')
            for i in trained:
                if i in Cluster.Clients:
                    print(i, end=', ')
            print()


        ## FedAvg聚合
        FedAvg_global_model.load_state_dict(avg(FedAvg_global_model.state_dict(), FedAvg_local_models))
        FedAvgTestDataLoader = datasetGen.get_fedavg_test_DataLoader()
        loss, acc = test(FedAvg_global_model, FedAvgTestDataLoader, device)
        FedAvg_Loss.append(loss)
        FedAvg_Acc.append(acc)
        TotalLoss.append(np.mean(epoch_loss))
        TotalAcc.append(np.mean(epoch_acc))
        print('acc_list : ', epoch_acc)
        print("Epoch: {}\t, FedAvg\t: Acc : {}\t, Loss : {}\t".format(epoch, FedAvg_Acc[epoch], FedAvg_Loss[epoch]))
        print("Epoch: {}\t, HCCFL\t: Acc : {}\t, Loss : {}\t".format(epoch, TotalAcc[epoch], TotalLoss[epoch]))


    SavePath = args.save_path + 'NewData_round_100_WithTimeAvg_HCCFL_FedAvg_Loss_Acc_0'
    # torch.save(client_update_grad,
    #            SavePath + '.pt')
    torch.save(FedAvg_Loss,
               SavePath + '_FedAvg_Loss.pt')
    torch.save(FedAvg_Acc,
               SavePath + '_FedAvg_Acc.pt')
    # torch.save(client_update_grad_with_,
    #            SavePath + '_weighting_grad.pt')
    torch.save(TotalLoss,
               SavePath + '_HCCFL_Loss.pt')
    torch.save(TotalAcc,
               SavePath + '_HCCFL_Acc.pt')


def L2_Distance(tensor1, tensor2, Use_cos = False):

    if Use_cos:
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
    else:
        Value = 0
        for i in range(tensor1.shape[0]):
            Value += math.pow(tensor1[i].item() - tensor2[i].item(), 2)
        return Value


def avg_deep_param(model_dict, args):
    AvgParam = torch.zeros(model_dict[args.deep_model_layer_name].shape[0])
    for i in range(model_dict[args.deep_model_layer_name].shape[1]):
        for j in range(model_dict[args.deep_model_layer_name].shape[0]):
            AvgParam[j] = AvgParam[j] + (model_dict[args.deep_model_layer_name][j][i])
    return AvgParam / model_dict[args.deep_model_layer_name].shape[1]


def avg_deep_param_with_dir(model_dict, pre_model_dict, args):
    AvgParam = torch.zeros(model_dict[args.deep_model_layer_name].shape[0])
    for i in range(model_dict[args.deep_model_layer_name].shape[1]):
        for j in range(model_dict[args.deep_model_layer_name].shape[0]):
            AvgParam[j] = AvgParam[j] + (model_dict[args.deep_model_layer_name][j][i] - pre_model_dict[args.deep_model_layer_name][j][i])
    return AvgParam / model_dict[args.deep_model_layer_name].shape[1]


def update_client_similarity_matrix(clients_model: Dict[int, ClientInServerData], args):
    similarity_matrix = {client_id_l:{client_id_r: 0.0 for client_id_r in clients_model.keys()} for client_id_l in clients_model.keys()}
    for client_id_l, Client_l in clients_model.items():
        client_l_avg_param = avg_deep_param(Client_l.PreModelStaticDict, args)
        for client_id_r, Client_r in clients_model.items():
            client_r_avg_param = avg_deep_param(Client_r.PreModelStaticDict, args)
            similarity_matrix[client_id_l][client_id_r] = L2_Distance(client_l_avg_param, client_r_avg_param)
    return similarity_matrix


def calculate_relative_similarity(clients_model, global_model, round_clients, old_matrix, args):
    print('计算相似度')
    similarity_matrix = {client_id_l: {client_id_r: 0.0 for client_id_r in clients_model.keys()} for client_id_l in
                         clients_model.keys()}
    for client_id_l, dis_l_dict in old_matrix.items():
        for client_id_r, dis_ in dis_l_dict.items():
            similarity_matrix[client_id_l][client_id_r] = dis_

    for client_id_l, Client_l in clients_model.items():
        if client_id_l in round_clients:
            client_l_avg_param = avg_deep_param_with_dir(Client_l.ModelStaticDict, global_model.state_dict(), args)
            # client_l_model_dict = Client_l.ModelStaticDict
            for client_id_r, Client_r in clients_model.items():
                client_r_avg_param = avg_deep_param_with_dir(Client_r.ModelStaticDict, global_model.state_dict(), args)
                # client_r_model_dict = Client_r.ModelStaticDict
                # min_dis = calculate_min_dis(client_l_model_dict, client_r_model_dict, Client_l.PreModelStaticDict, Client_r.PreModelStaticDict, args)
                Dis = L2_Distance(client_l_avg_param, client_r_avg_param, True)
                similarity_matrix[client_id_l][client_id_r] = Dis
                similarity_matrix[client_id_r][client_id_l] = Dis

    return similarity_matrix


def calculate_min_dis(model_dict_1, model_dict_2, client_l_pre_dict, client_r_pre_dict, args):
    min_dis = 2.0
    for i in range(model_dict_1[args.deep_model_layer_name].shape[0]):
        model_deep_1 = model_dict_1[args.deep_model_layer_name][i][:] - client_l_pre_dict[args.deep_model_layer_name][i][:]
        model_deep_2 = model_dict_2[args.deep_model_layer_name][i][:] - client_r_pre_dict[args.deep_model_layer_name][i][:]

        dis = L2_Distance(model_deep_1, model_deep_2)
        if dis < min_dis:
            min_dis = dis

    return min_dis

    #
    # for client_id_l, clients in similarity_matrix.items():
    #     client_l_deep_avg_param = avg_deep_param(clients_model[client_id_l].ModelStaticDict)
    #     for client_id_r, SimValue in clients.items():
    #         client_r_deep_avg_param = avg_deep_param(clients_model[client_id_r].ModelStaticDict)
    #         similarity_matrix[client_id_l][client_id_r] = L2_Distance(client_l_deep_avg_param, client_r_deep_avg_param)


def init_test_dataset_loader(dataset_name, batch_size):
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        pass


def save(global_test_eval, global_cost, model_dict, args):
    args.to_string('FedPro')
    dir_path = args.save_path + '/' + 'Experiment'
    dir_path_id = 0
    while True:
        save_path = dir_path + str(dir_path_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            break
        else:
            dir_path_id += 1

    torch.save(global_cost,
               save_path + '/' + 'Global_Cost.pt')
    torch.save(global_test_eval,
               save_path + '/' + 'Global.pt')
    torch.save(model_dict,
               save_path + '/' + 'Model_Dict.pt')

    f = open(save_path + '/实验描述', 'w', encoding='UTF-8')
    f.write(args.Arg_string)
    f.write('\n' + str(datetime.datetime.now()))
    f.close()


if __name__ == '__main__':
    MyArgs = Args.Arguments()
    main(MyArgs)
