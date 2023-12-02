import copy
from collections import OrderedDict
import json



class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.local_epochs = 2
        self.lr = 0.0005
        self.save_model = True
        self.global_round = 300
        # 'cifar10'   'mnist'
        self.dataset_name = 'cifar10'
        if self.dataset_name == 'mnist':
            self.deep_model_layer_name = 'fc4.weight'
            self.model_name = "mnist"
        else:
            self.deep_model_layer_name = 'fc3.weight'  # 'fc3.weight'
            self.model_name = "NewCifar10"
        self.dataset_labels_number = 10
        self.worker_num = 100
        self.test_worker_num = 200
        self.worker_train = 10
        self.cluster_worker_train = 2
        self.MaxProcessNumber = 3
        self.optim = 'Adam' # 'SGD', 'Adam'

        # 每个本地客户端分配的数据数量
        self.local_data_size = 0.4
        self.rot_local_data_size = 200
        self.data_classes_dict = {'cifar10': 10, 'mnist': 10}
        self.data_classes = self.data_classes_dict[self.dataset_name]
        # 每个本地客户端分配的数据标签类别占比
        self.local_data_classes = 0.4
        # 数据分布
        self.dataset_labels_num = 10
        self.data_info = {'data_labels': [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [0, 1, 8, 9]],
                          # [[0,1,2,3,4,5,6,7,8,9]]
                          # Distribution-Parallel     [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
                          # Distribution-Cover        [[0, 1, 2, 3, 4], [1, 2, 3], [5, 6, 7, 8, 9], [6, 7, 8], [0, 4, 5, 9]]
                          # Distribution-Overlapping  [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [0, 1, 8, 9]]
                          'data_rot': [0, 1, 2, 3],
                          'divide_type': 'labels'
                          }# 'rot'

        self.cluster_number = len(self.data_info["data_labels"]) if self.data_info["divide_type"]=="labels" else len(self.data_info["data_rot"])
        self.cuda = True
        self.save_path = ""
        if len(self.data_info) > 0:
            self.save_path = self.get_file_name() + '_cluster_self_data' + str(self.worker_train)+"_"+str(self.cluster_worker_train)
        else:
            self.save_path = self.get_file_name() + '_cluster_' + str(self.worker_train) + "_" + str(self.cluster_worker_train)
        self.Arg_string = ''
        self.kMeansArgs = KMeansArgs()

    def get_file_name(self):
        return str(self.local_data_size)+'_'+str(self.local_data_classes) + '_' + self.dataset_name

    def list_all_member(self):
        text = ''
        for name, value in vars(self).items():
            text += str(name) + ":  " + str(value) + '\n'

        return text

    def to_string(self, text=''):
        self.Arg_string = '实验描述：\n' + text + '\n' + '**************************\n' + self.list_all_member()

    def save_dict(self):
        str_dict = {'algorithm_name': "", 'acc': 0, 'loss': 0, 'traffic': 0,'final_cluster_number': 0,'extra_param':"",
            'batch_size': self.batch_size, 'local_epoch': self.local_epochs, 'lr': self.lr,
                    'global_round': self.global_round, 'dataset': self.dataset_name, 'model': self.model_name,
                    'worker_num': self.worker_num, 'worker_train': self.worker_train,
                    'deep_model_name': self.deep_model_layer_name, 'optim': self.optim,
                    'data_type': self.data_info["divide_type"],
                    'data_info': self.data_info["data_labels"] if self.data_info["divide_type"] == "labels" else
                    self.data_info["data_rot"],
                    'acc_list': [],
                    'loss_list': []}
        return str_dict

    def quant_save_dict(self):
        str_dict = {
            "algorithm_name": "",
            'acc': 0,
            'loss': 0,
            'extra_param':"",
            "acc_list":[],
            "loss_list":[],
            'data_info': self.data_info["data_labels"] if self.data_info["divide_type"] == "labels" else self.data_info["data_rot"],
            'dataset': self.dataset_name,
            'model': self.model_name,
            'batch_size': self.batch_size,
            'local_epoch': self.local_epochs,
            'lr': self.lr,
            'optim': self.optim,
        }
        return str_dict





class KMeansArgs:
    def __init__(self):
        self.K = 5
        self.max_iter = 50
        self.local_epoch = 5


class LocalModelList:
    def __init__(self):
        self.ModelDictList: OrderedDict = OrderedDict()

    def add_model_dict(self, model_dict, round):
        if len(self.ModelDictList.keys()) == 22:
            self.ModelDictList.popitem(last=False)
        self.ModelDictList[round] = model_dict


class ClientInServerData:
    def __init__(self, ID: int, ModelDict, inClusterID, Round):
        self.ClientID: int = ID
        self.InClusterID = inClusterID
        self.ModelStaticDict = ModelDict

        self.PreModelStaticDict = None
        self.LocalDictUpdate = None

        self.LocalToGlobalResDictUpdate = None
        self.LocalToGlobalResRound = 0
        self.DataLen = 0
        # 描述当前客户端最新的参与训练的轮次
        self.TrainRound = Round
        self.NumRounds = 0
        # 记录客户端的历史训练的本地模型更新模型
        self.LocalModelUpdateList: LocalModelList = LocalModelList()

    def set_client_info(self, ModelDict, DataSize, local_res_dict):
        self.ModelStaticDict = copy.deepcopy(ModelDict)
        self.DataLen = DataSize
        self.LocalResDictUpdate = local_res_dict

    def add_model_update(self, mode_dict, t_round):
        self.LocalModelUpdateList.add_model_dict(mode_dict, t_round)


    def set_client_InClusterID(self, ClusterID):
        self.InClusterID = ClusterID


    def update_global_res(self):
        self.LocalToGlobalResDictUpdate = copy.deepcopy(self.LocalResDictUpdate)


class AlgorithmParams:
    def __init__(self):
        self.random_seed= 2,
        # 是否使用本地量化
        self.is_quant_local_update= True

        # 是否使用 上传的 残差bit量化
        self.is_quant_up_local_res= True
        self.is_quant_down_global_res= True

        # 是否将全局模型更新进行量化后 发送给客户端
        self.is_quant_down_update= True

        # 是否使用全局残差同步
        self.is_ues_global_res= True
        # 是否将全局残差直接加到全局模型 并发送给客户端
        self.is_add_global_res_to_model= True

        self.spare_rate= 0.1
        self.pre_global_res_update_round= 10

        self.quanter_name = "STCQuanter"

class ArgsSet:
    def __init__(self, config_name, data_name):
        self.DataJsonObject = None
        self.ConfigJsonObject = None
        self.ConfigJsonObject, self.DataJsonObject = self.read_json(config_name, data_name)


    def get_Description(self):
        return self.ConfigJsonObject["Description"]

    def read_json(self, config_name, data_name):
        with open('package.json', encoding='utf-8') as f:
            superHeroSquad = json.load(f)
        return superHeroSquad[config_name], superHeroSquad[data_name]

    def set_all_args(self, args: Arguments, clusterManager, algorithm_: AlgorithmParams):
        self.set_Argument_args(args)
        self.set_Cluster_args(clusterManager)
        self.set_Algorithm_args(algorithm_)

    def set_Argument_args(self, args: Arguments):
        Argument_args = self.DataJsonObject
        args.lr = Argument_args["lr"]
        args.global_round = Argument_args["global_round"]
        args.dataset_name = Argument_args["dataset_name"]

        if args.dataset_name == 'mnist':
            args.deep_model_layer_name = 'fc4.weight'
            args.model_name = "mnist"
        else:
            args.deep_model_layer_name = 'fc3.weight'  # 'fc3.weight'
            args.model_name = "NewCifar10"

        args.optim = Argument_args["optim"]
        args.data_info["data_labels"] = Argument_args["data_info"]["data_labels"]
        args.data_info["data_rot"] = Argument_args["data_info"]["data_rot"]
        args.data_info["divide_type"] = Argument_args["data_info"]["divide_type"]

        args.cluster_number = len(args.data_info["data_labels"]) if args.data_info["divide_type"] == "labels" else len(
            args.data_info["data_rot"])
        args.cuda = Argument_args["cuda"]

    def set_Cluster_args(self, clusterManager):
        Cluster_args = self.DataJsonObject["ClusterTree"]
        clusterManager.H = Cluster_args["H"]

    def set_Algorithm_args(self, algorithm_: AlgorithmParams):
        Argument_args = self.ConfigJsonObject["AlgorithmParams"]
        algorithm_.random_seed = Argument_args["random_seed"]

        algorithm_.is_quant_local_update = Argument_args["is_quant_local_update"]

        algorithm_.is_quant_up_local_res = Argument_args["is_quant_up_local_res"]
        algorithm_.is_quant_down_global_res = Argument_args["is_quant_down_global_res"]
        algorithm_.is_quant_down_update = Argument_args["is_quant_down_update"]
        algorithm_.is_ues_global_res = Argument_args["is_ues_global_res"]
        algorithm_.is_add_global_res_to_model = Argument_args["is_add_global_res_to_model"]

        algorithm_.spare_rate = Argument_args['spare_rate']
        algorithm_.pre_global_res_update_round = Argument_args['pre_global_res_update_round']

        algorithm_.quanter_name = Argument_args['quanter_name']
