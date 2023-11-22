
from DataGenerater import *
import HCClusterTree
import random
import copy
import math
import Args
from Args import ClientInServerData
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import torch
import Model
import time
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from HCFedAvg import FileProcess
spare_rate = 0.3
pre_global_res_update_round = 10

def train(global_model_dict, datasetLoader, worker_id, device, args, res_model_dict=None, use_res = True, is_quant = True):
    # 创建量化器
    if is_quant:
        quanter = Model.SpareBinaryQuanter()
        quanter.set_spare_rate(spare_rate)

    # 创建模型
    local_model = Model.init_model(args.model_name)
    local_model.load_state_dict(global_model_dict)

    # 为模型设置量化器
    if is_quant:
        local_model.set_quanter(quanter)

    old_model_dict = copy.deepcopy(global_model_dict)

    optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
    local_model.train()
    local_model.to(device=device)
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

            pred = local_model(batch_data)
            # print(pred)
            loss = F.nll_loss(pred, batch_label)

            avg_loss += loss.item()
            count +=1

            loss.backward()

            optimizer.step()
    # print(worker_id)
    # print(avg_loss)
    local_model.to('cpu')
    # print(avg_loss/count)
    data_len = data_size
    cost = 0

    new_model_dict = local_model.state_dict()

    update_model_dict = copy.deepcopy(new_model_dict)

    for name, param in update_model_dict.items():
        update_model_dict[name] = update_model_dict[name] - old_model_dict[name]

    # return update_model_dict, data_len, res_model_dict
    if is_quant:
        if use_res:

            if res_model_dict is not None and use_res:
                for name, parma in update_model_dict.items():
                    update_model_dict[name] = parma + res_model_dict[name]
            else:
                res_model_dict = copy.deepcopy(update_model_dict)

        quanted_model_dict = local_model.Quanter.quant_model(update_model_dict)

        if use_res:
            for name, param in quanted_model_dict.items():
                res_model_dict[name] = update_model_dict[name] - quanted_model_dict[name]

        return {'data_len': data_len,
               'id': worker_id,
               'cost': cost * 4,
               'quanted_model_update':quanted_model_dict,
                'res_model_update': res_model_dict,
                'model_update': None}

    return {'data_len': data_len,
               'id': worker_id,
               'cost': cost * 4,
               'quanted_model_update':None,
                'res_model_update': res_model_dict,
                'model_update': update_model_dict,
                'model': new_model_dict}
    # loss, acc = test(model, test_loader, device)
    # q.put()
    # shared_models[worker_id]['model'].load_state_dict(copy.deepcopy(model.state_dict()))

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

def main(args):

    datasetGen = DatasetGen(args)

    train_workers = [i for i in range(args.worker_num)]

    random_seed = 2
    # FedAvg算法的模型
    torch.manual_seed(random_seed)

    FedAvg_global_model = Model.init_model(args.model_name)

    # 每个客户端的 局部模型，初始化时为相同的模型

    global_model = Model.init_model(args.model_name)

    FedAvg_global_model.load_state_dict(global_model.state_dict())

    clients_model: Dict[int, ClientInServerData] = {}

    ClusterManager = HCClusterTree.HCClusterManager()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    TotalLoss = []
    TotalAcc = []
    FinalClusterNumber = []
    SimMean = []
    SimSTD = []

    ### 加权随机
    clients_time = [1 for i in range(args.worker_num)]

    ###

    old_matrix = {}
    current_max_acc = 0

    is_set_cluster_res = False
    global_res_round = 0

    is_quant = True
    for epoch in range(args.global_round):

        ### 加权随机
        e_weights = [math.pow(math.e, i) for i in clients_time]
        sum = np.sum(e_weights)
        e_weights_1 = [i / sum for i in e_weights]
        cluster_clients_train = np.random.choice(a=train_workers, size=args.worker_train, replace=False, p=e_weights_1)

        for k in train_workers:
            if k in cluster_clients_train:
                clients_time[k] = 1
            else:
                clients_time[k] += 1
        ###
        ## 纯随机
        # cluster_clients_train = random.sample(train_workers, args.worker_train)
        ###

        # for c in cluster_clients_train:
        #     if c in train_workers:
        #         train_workers.remove(c)
        #
        # if len(train_workers) == 0:
        #     train_workers = [i for i in range(args.worker_num)]

        print(' 参与训练的客户端 ：')
        print(cluster_clients_train)

        for i, worker_id in enumerate(cluster_clients_train):
            res_dict = None

            if clients_model.get(worker_id) is None:
                clients_model[worker_id] = ClientInServerData(worker_id, global_model.state_dict(), worker_id, 0)
                train_model_dict = global_model.state_dict()
            else:
                current_cluster = ClusterManager.get_cluster_by_id(clients_model[worker_id].InClusterID)
                train_model_dict = current_cluster.get_avg_cluster_model_copy()
                # 如果 本地残差比 全局的新，用本地的， 如果全局残差为空，也用本地的
                if is_quant:
                    if clients_model[worker_id].TrainRound >= current_cluster.GlobalResRound:
                        res_dict = clients_model[worker_id].LocalResDictUpdate
                    else:
                        if current_cluster.get_avg_cluster_res_copy() is not None:
                            print("全局残差")
                            res_dict = current_cluster.get_avg_cluster_res_copy()
                            train_model_dict = model_add(train_model_dict, res_dict)
                            res_dict = None
                        else:
                            res_dict = clients_model[worker_id].LocalResDictUpdate

                clients_model[worker_id].TrainRound = epoch

            clients_model[worker_id].PreModelStaticDict = copy.deepcopy(train_model_dict)

            train_info = train(train_model_dict, datasetGen.get_client_DataLoader(worker_id), worker_id, device, args, res_dict, True, is_quant)

            if is_quant:
                local_model = model_add(train_model_dict, train_info['quanted_model_update'])
            else:
                local_model = model_add(train_model_dict, train_info['model_update'])

            clients_model[worker_id].set_client_info(local_model, train_info['data_len'], train_info['res_model_update'])

        # global_si_ma = calculate_similarity(clients_model, global_model, cluster_clients_train, old_matrix, args)
        global_si_ma = calculate_relative_similarity(clients_model, global_model, cluster_clients_train, old_matrix, args)
        # global_si_ma = calculate_sim_only_cos(clients_model, global_model, cluster_clients_train, old_matrix, args)

        old_matrix = copy.deepcopy(global_si_ma)

        ClusterManager.reset_similarity_matrix(global_si_ma)
        # std_m = []
        # td_sm = copy.deepcopy(global_si_ma)
        # for key, value in td_sm.items():
        #     value.pop(key)
        #     std_m.extend(value.values())
        # std = np.mean(std_m)
        # print('mean: ', std)
        t1 = time.time()
        ClusterManager.HCClusterDivide()
        t2 = time.time()

        print("Clustering Time: ", t2-t1)

        # 消融实验
        mean_, std_ = ClusterManager.calculate_clusters_sd()
        SimMean.append(mean_)
        SimSTD.append(std_)


        # ClusterManager.print_divide_result()

        # ClusterManager.UpdateClusterAvgModel(clients_model, cluster_clients_train)
        t1 = time.time()

        if is_quant:
            if epoch % pre_global_res_update_round == 0 and epoch != 0:
                global_res_round = epoch
                # 更新本地 残差上传到服务端 的 记录
                for worker_id in cluster_clients_train:
                    clients_model[worker_id].LocalToGlobalResRound = global_res_round
                    clients_model[worker_id].update_global_res()

        # 更新本地 上传到服务端的 残差记录
        ClusterManager.UpdateClusterAvgModelAndResWithTime(clients_model, is_quant)
        t2 = time.time()
        print("Avg Time: ", t2 - t1)
        epoch_loss = []
        epoch_acc = []

        top_acc = [0 for i in range(args.cluster_number)]

        ## 集群准确性测试
        for cluster_id, Cluster in ClusterManager.CurrentClusters.items():
            test_dataloader = datasetGen.get_cluster_test_DataLoader(cluster_id % args.cluster_number)
            test_model = Model.init_model(args.model_name)
            test_model.load_state_dict(Cluster.get_avg_cluster_model_copy())
            loss, acc = test(test_model, test_dataloader, device)
            epoch_loss.append(loss)
            epoch_acc.append(acc)

            if top_acc[int(cluster_id) % args.cluster_number] < acc:
                top_acc[int(cluster_id % args.cluster_number)] = acc

        # 输出当前轮次集群结果

        # trained = cluster_clients_train[:]
        # print(' 轮次划分结果 ')
        # for cluster_id, Cluster in ClusterManager.CurrentClusters.items():
        #     print('cluster_id: ', cluster_id, ' , res: ', end='')
        #     for i in trained:
        #         if i in Cluster.Clients:
        #             print(i, end=', ')
        #     print()

        sorted_top_acc = sorted(top_acc, reverse= True)[:len(ClusterManager.CurrentClusters)]
        FinalClusterNumber.append(len(ClusterManager.CurrentClusters))
        TotalLoss.append(np.mean(sorted(epoch_loss, reverse=True)[:5]))
        TotalAcc.append(np.mean(sorted_top_acc))
        print('acc_list : ', epoch_acc)
        print("top_acc", sorted_top_acc)
        print("mean_acc", np.mean(sorted_top_acc))
        if TotalAcc[epoch] > current_max_acc:
            current_max_acc = TotalAcc[epoch]
        print("Epoch------------------------------------: {}\t, HCCFL\t: Acc : {}\t, Max_Acc : {}\t".format(epoch, TotalAcc[epoch], current_max_acc))

    save_dict = args.save_dict()
    save_dict['algorithm_name'] = 'HCCFL_res_spare_0.3_res_5'
    save_dict['acc'] = max(TotalAcc)
    save_dict['loss'] = min(TotalLoss)
    save_dict['traffic'] = 200*10
    save_dict['acc_list'] = TotalAcc
    save_dict['loss_list'] = TotalLoss
    save_dict['extra_param'] = "random seed " + str(random_seed) + " H " + str(ClusterManager.H)
    save_dict['final_cluster_number'] = FinalClusterNumber
    save_dict['sim_std'] = SimSTD
    save_dict['sim_mean'] = SimMean

    FileProcess.add_row(save_dict)


def model_add(global_model_dict, add_model_dict):
    copy_model_dict = copy.deepcopy(global_model_dict)
    for key in copy_model_dict.keys():
        copy_model_dict[key] += add_model_dict[key]
    return copy_model_dict


def L2_Distance(tensor1, tensor2, Use_cos = False, Use_L2 = False):

    if Use_L2:
        sum_ = 0
        for i in range(tensor1.shape[0]):
            sum_ +=  math.pow(tensor1[i].item() - tensor2[i].item(), 2)
        sum_ = math.pow(sum_, 0.5)
        return sum_


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

        if DownSum2 == 0 or DownSum1 == 0:
            return 0

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


def trans_param_to_tensor(model_dict):

    parameters = [param.data.view(-1) for param in model_dict.values()]
    m_parameters = torch.cat(parameters)

    return m_parameters

def dt_matrix(param_list):
    a = b = torch.stack(param_list, dim=0)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res

def calculate_sim_only_cos(clients_model: Dict[int, ClientInServerData], global_model, round_clients, old_matrix, args):
    print('计算相似度')
    similarity_matrix = {client_id_l: {client_id_r: 0.0 for client_id_r in clients_model.keys()} for client_id_l in
                         clients_model.keys()}
    for client_id_l, dis_l_dict in old_matrix.items():
        for client_id_r, dis_ in dis_l_dict.items():
            similarity_matrix[client_id_l][client_id_r] = dis_


    for client_id_l, Client_l in clients_model.items():
        if client_id_l in round_clients:
            client_l_avg_param = avg_deep_param_with_dir(Client_l.ModelStaticDict, Client_l.PreModelStaticDict, args)
            # client_l_model_dict = Client_l.ModelStaticDict
            for client_id_r, Client_r in clients_model.items():
                client_r_avg_param = avg_deep_param_with_dir(Client_r.ModelStaticDict, Client_r.PreModelStaticDict, args)
                # client_r_model_dict = Client_r.ModelStaticDict
                # min_dis = calculate_min_dis(client_l_model_dict, client_r_model_dict, Client_l.PreModelStaticDict, Client_r.PreModelStaticDict, args)
                Dis = L2_Distance(client_l_avg_param, client_r_avg_param, True)
                similarity_matrix[client_id_l][client_id_r] = Dis
                similarity_matrix[client_id_r][client_id_l] = Dis

    return similarity_matrix

# TAS
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
            client_l_pre_param = avg_deep_param_with_dir(Client_l.PreModelStaticDict, global_model.state_dict(), args)

            for client_id_r, Client_r in clients_model.items():
                ex_dis = 0
                client_r_avg_param = avg_deep_param_with_dir(Client_r.ModelStaticDict, global_model.state_dict(), args)
                client_r_pre_param = avg_deep_param_with_dir(Client_r.PreModelStaticDict, global_model.state_dict(), args)

                all_dis = L2_Distance(client_l_avg_param, client_r_avg_param, True)
                if Client_l.InClusterID == Client_r.InClusterID:
                    ex_dis = L2_Distance(client_l_pre_param, client_r_pre_param, True)

                Dis = abs(ex_dis-all_dis)
                similarity_matrix[client_id_l][client_id_r] = Dis
                similarity_matrix[client_id_r][client_id_l] = Dis

    return similarity_matrix

# 使用纯 cos 计算相似度
def calculate_relative_similarity_cos(clients_model, round_clients, old_matrix, args):
    print('计算相似度')
    similarity_matrix = {client_id_l: {client_id_r: 0.0 for client_id_r in clients_model.keys()} for client_id_l in
                         clients_model.keys()}
    for client_id_l, dis_l_dict in old_matrix.items():
        for client_id_r, dis_ in dis_l_dict.items():
            similarity_matrix[client_id_l][client_id_r] = dis_

    for client_id_l, Client_l in clients_model.items():
        if client_id_l in round_clients:
            client_l_avg_param = avg_deep_param(Client_l.ModelStaticDict, args)

            for client_id_r, Client_r in clients_model.items():

                client_r_avg_param = avg_deep_param(Client_r.ModelStaticDict, args)
                all_dis = L2_Distance(client_l_avg_param, client_r_avg_param, True)

                similarity_matrix[client_id_l][client_id_r] = all_dis
                similarity_matrix[client_id_r][client_id_l] = all_dis

    return similarity_matrix

# 使用纯 L2 计算相似度
def calculate_relative_similarity_L2(clients_model, round_clients, old_matrix, args):
    print('计算相似度')
    similarity_matrix = {client_id_l: {client_id_r: 0.0 for client_id_r in clients_model.keys()} for client_id_l in
                         clients_model.keys()}
    for client_id_l, dis_l_dict in old_matrix.items():
        for client_id_r, dis_ in dis_l_dict.items():
            similarity_matrix[client_id_l][client_id_r] = dis_

    for client_id_l, Client_l in clients_model.items():
        if client_id_l in round_clients:
            client_l_avg_param = avg_deep_param(Client_l.ModelStaticDict, args)

            for client_id_r, Client_r in clients_model.items():

                client_r_avg_param = avg_deep_param(Client_r.ModelStaticDict, args)
                all_dis = L2_Distance(client_l_avg_param, client_r_avg_param, False, True)

                similarity_matrix[client_id_l][client_id_r] = all_dis
                similarity_matrix[client_id_r][client_id_l] = all_dis

    return similarity_matrix


def calculate_similarity(clients_model, global_model, round_clients, old_matrix, args):
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
        dataset = datasets.MNIST(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='data', train=False, transform=transforms.Compose([
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
