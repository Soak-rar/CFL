import torch
import random
import Args
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


def train(global_model_state_dict, dataset_dict, worker_id, device, epoch, delay, args):
    model = Model.init_model(args.model_name)
    model_global = Model.init_model(args.model_name)
    model.load_state_dict(global_model_state_dict)
    model_global.load_state_dict(global_model_state_dict.copy())

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    mmd_optimizer = optim.SGD(model.fc4.parameters(), lr=args.lr)

    train_loader = init_local_dataloader(dataset_dict, args)

    model.train()
    model.to(device)
    model_global.to(device)
    sum_mmd_loss = 0
    for local_epoch in range(args.local_epochs):
        for batch_index, (batch_data, batch_label) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_data)

            loss = F.nll_loss(pred, batch_label)

            loss.backward()

            optimizer.step()
            if delay:
                mmd_optimizer.zero_grad()

                pred = model(batch_data)
                mmd_loss = MMDLoss.data_mmd(model_global.fc4.state_dict()['weight'], model.fc4.state_dict()['weight'])
                mmd_loss.requires_grad = True
                sum_mmd_loss += mmd_loss
                mmd_loss = mmd_loss

                mmd_loss.backward()

                mmd_optimizer.step()
    del model_global
    # loss, acc = test(model, test_loader, device)
    # print('Global_epoch: {}  , loss : {} , acc : {}'.format(epoch, loss, acc))
    model.to('cpu')
    return {'model_dict': model.state_dict(),
            'data_len': dataset_dict['data_len'],
            'id': worker_id}


def init_local_dataloader(dataset_dict, args):
    train_dataset = TensorDataset(dataset_dict['data'],
                                  dataset_dict['label'].to(torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader


def test(model, dataset_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    model.to(device)
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
    train_workers = Data.load_data(args)
    global_model = Model.init_model(args.model_name)
    test_dataset_loader = init_test_dataset_loader(args.dataset_name, args.test_batch_size)

    global_loss_list = []
    global_acc_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    start_time = time.time()
    do_mmd = False
    for epoch in range(args.global_round):
        train_workers_id = random.sample(range(args.worker_num), args.worker_train)
        local_models = []
        if epoch == 3:
            do_mmd = True
        for worker_id in train_workers_id:
            train_eval = train(global_model.state_dict().copy(), train_workers[worker_id], worker_id, device, epoch, do_mmd, args)
            local_models.append(train_eval)

        avg_model_dict = avg(global_model.state_dict(), local_models)
        global_model.load_state_dict(avg_model_dict)

        global_loss, global_acc = test(global_model, test_dataset_loader, device)
        global_loss_list.append(global_loss)
        global_acc_list.append(global_acc)

        epoch_time = time.time()
        # 输出一次epoch的指标
        print('Global_Epoch: {}  ,  Loss: {:.5f},  Acc: {:.4f},  Epoch_Total_Time: {}min {:.2f}second\n'
              .format(epoch + 1, global_loss, global_acc * 100, int((epoch_time - start_time) / 60), (epoch_time - start_time) % 60))

    global_test_eval = {'acc': global_acc_list, 'loss': global_loss_list}

    save(global_test_eval, global_model.state_dict(), args)


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


def save(global_test_eval, model_dict, args):
    args.to_string('MMDBack： loss 进行 指定层参数反向传播训练 ，mmd_loss 的权重为100%， 延迟3轮进行mmd\n将参与训练的客户端设为40%')
    dir_path = args.save_path + '/' + '实验'
    dir_path_id = 0
    while True:
        save_path = dir_path + str(dir_path_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            break
        else:
            dir_path_id += 1

    torch.save(global_test_eval,
               save_path + '/' + 'Global.pt')
    torch.save(model_dict,
               save_path + '/' + 'Model_Dict.pt')

    f = open(save_path + '/实验描述', 'w', encoding='UTF-8')
    f.write(args.Arg_string)
    f.close()


if __name__ == '__main__':
    args = Args.Arguments()
    main(args)
