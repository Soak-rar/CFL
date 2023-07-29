import copy


class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.local_epochs = 1
        self.lr = 0.0005
        self.save_model = True
        self.global_round = 200
        # 'cifar10'   'mnist'
        self.dataset_name = 'cifar10'
        self.model_name = "NewCifar10"
        self.dataset_labels_number = 10
        self.worker_num = 100
        self.test_worker_num = 200
        self.worker_train = 10
        self.cluster_worker_train = 2
        self.deep_model_layer_name = 'fc3.weight'
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
        self.data_info = {'data_labels': [[0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9], [0,1,8,9]],
                          'data_rot': [0, 1, 2, 3],
                          'divide_type': 'labels'}# 'rot'

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
        str_dict = {'algorithm_name': "", 'acc': 0, 'loss': 0, 'traffic': 0,
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





class KMeansArgs:
    def __init__(self):
        self.K = 5
        self.max_iter = 50
        self.local_epoch = 5


class ClientInServerData:
    def __init__(self, ID: int, ModelDict, inClusterID, Round):
        self.ClientID: int = ID
        self.InClusterID = inClusterID
        self.ModelStaticDict = ModelDict
        self.PreModelStaticDict = None
        self.DataLen = 0
        # 描述当前客户端最新的参与训练的轮次
        self.TrainRound = Round
        self.NumRounds = 0

    def set_client_info(self, ModelDict, DataSize):
        self.ModelStaticDict = copy.deepcopy(ModelDict)
        self.DataLen = DataSize


    def set_client_InClusterID(self, ClusterID):
        self.InClusterID = ClusterID


