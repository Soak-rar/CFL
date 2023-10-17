## 核心思路：指定 K 个集群，每次循环将 K 个集群的模型 全部发送给 参与训练的客户端， 客户端训练 K 个模型，
## 并根据 K 模型的损失 将 客户端指定为 损失最小的模型对应的集群中
import copy

import torch
from tqdm import tqdm
from HCFedAvg.DataGenerater import *
import Args
import Model
import torch.nn.functional as F
import torch.optim as optim
from HCFedAvg import FileProcess
import global_set


spare_rate = 0.3

def train(global_model_dict, datasetLoader, worker_id,res_model_dict ,device, args: Args.Arguments, client_time,use_res = True):
    # if res_model_dict is not None and use_res:
    #     for name, parma in global_model_dict.items():
    #         global_model_dict[name] = parma + res_model_dict[name]
    # else:
    #     res_model_dict = copy.deepcopy(global_model_dict)


    quanter = Model.SpareBinaryQuanter()
    quanter.set_spare_rate(spare_rate)
    local_model = Model.init_model(args.model_name)



    local_model.load_state_dict(global_model_dict)

    local_model.set_quanter(quanter)

    old_model_dict = copy.deepcopy(global_model_dict)

    if args.optim == 'Adam':
        optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(local_model.parameters(), lr=args.lr)

    local_model.to(device=device)
    local_model.train()
    loss_count = 0
    loss_sum = 0.0
    batch_num = 0

    for local_epoch in range(args.local_epochs):
        for batch_index, (batch_data, batch_label) in enumerate(datasetLoader):
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = local_model(batch_data)
            loss = F.nll_loss(pred, batch_label)
            loss_sum += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()

            if local_epoch == 0:
                batch_num += 1

    data_len = batch_num * args.batch_size
    local_model.to('cpu')
    print('local  ', worker_id, ',  loss  ', loss_sum/loss_count)

    new_model_dict = local_model.state_dict()

    update_model_dict = copy.deepcopy(new_model_dict)

    for name, param in update_model_dict.items():
        update_model_dict[name] = update_model_dict[name] - old_model_dict[name]

    # return update_model_dict, data_len, res_model_dict

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


    print("----------------------------, ", worker_id)

    # for name, param in update_model_param.items():
    #     print(name)
    #     print(param - pre_dict[name])



    return quanted_model_dict, data_len, res_model_dict



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


def main(mArgs):
    dataGen = DatasetGen(mArgs)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    torch.manual_seed(5)

    clients_time = [1 for i in range(args.worker_num)]

    global_model = Model.init_model(mArgs.model_name)

    train_workers = [i for i in range(args.worker_num)]

    TotalLoss = []
    TotalAcc = []

    res_model_clients = {i: None for i in range(args.worker_num)}

    latest_res = None
    is_Global_res = None

    for global_round in range(mArgs.global_round):
        worker_model_dicts = {}
        worker_data_len = {}

        clients_train = random.sample(train_workers, mArgs.worker_train)

        for k in train_workers:
            if k in clients_train:
                clients_time[k] = 1
            else:
                clients_time[k] += 1

        for worker_id in tqdm(clients_train, unit="client", leave=True):


            local_model, data_len, res_model_dict = train(copy.deepcopy(global_model.state_dict()),
                                                              dataGen.get_client_DataLoader(worker_id), worker_id,
                                                              latest_res, device, mArgs,
                                                              clients_time[worker_id], use_res=True)

            # print('L2  ',torch.norm(torch.tensor(local_model.Quanter.res), p=2))
            # dequant_model = local_model.dequant()
            # local_model.quant()
            # worker_model_dicts[worker_id] = local_model.dequant()
            res_model_clients[worker_id] = res_model_dict
            worker_model_dicts[worker_id] = local_model

            # worker_model_dicts[worker_id] = local_model.state_dict()
            worker_data_len[worker_id] = data_len

        data_sum = sum(worker_data_len.values())

        if global_round % 5 == 0:
            if latest_res is None:
                latest_res = copy.deepcopy(global_model.state_dict())
            for key in latest_res.keys():
                latest_res[key] *= 0
                for client_id in clients_train:
                    latest_res[key] += worker_model_dicts[client_id][key] * (
                         worker_data_len[client_id] * 1.0 / data_sum)
                # print(global_update_dict[key])


        global_dict = global_model.state_dict()
        global_update_dict = copy.deepcopy(global_dict)

        for key in global_update_dict.keys():
            global_update_dict[key] *= 0
            for client_id in clients_train:
                global_update_dict[key] += worker_model_dicts[client_id][key] * (worker_data_len[client_id] * 1.0 / data_sum)
            # print(global_update_dict[key])
            global_update_dict[key] += global_dict[key]



        global_model.load_state_dict(global_update_dict)
        test_dataloader = dataGen.get_fedavg_test_DataLoader()

        # 聚合残差

        loss, acc = test(global_model, test_dataloader, device)

        TotalAcc.append(acc)
        TotalLoss.append(loss)
        # print(cluster_id, "  test")
        print()
        print(" epoch :  ", global_round)
        print("acc ", acc)
        print('loss ', loss)


    save_dict = mArgs.quant_save_dict()
    save_dict['algorithm_name'] = 'FedAvg_Quant_without_res'
    save_dict['acc'] = max(TotalAcc)
    save_dict['loss'] = min(TotalLoss)
    save_dict['acc_list'] = TotalAcc
    save_dict['loss_list'] = TotalLoss
    save_dict['extra_param'] = "spare_rate " +  str(spare_rate)

    FileProcess.add_row_with_file_name(save_dict, global_set.ResultFileName)

def find_cluster_id(clients_list, cluster_num):
    clients_in_cluster = {cluster_id: 0 for cluster_id in range(cluster_num)}
    for client_id in clients_list:
        clients_in_cluster[client_id%cluster_num] += 1
    max_key = max(clients_in_cluster, key=clients_in_cluster.get)
    return max_key


if __name__ == '__main__':
    args = Args.Arguments()
    main(args)
