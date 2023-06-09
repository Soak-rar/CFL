import copy


class Arguments:
    def __init__(self):
        self.batch_size = 200
        self.test_batch_size = 64
        self.local_epochs = 2
        self.lr = 0.05
        self.save_model = True
        self.global_round = 100
        # 'cifar10'   'mnist'
        self.dataset_name = 'mnist'
        self.model_name = 'mnist'
        self.dataset_labels_number = 10
        self.worker_num = 100
        self.test_worker_num = 200
        self.worker_train = 10
        self.cluster_worker_train = 2
        # 每个本地客户端分配的数据数量
        self.local_data_size = 0.4
        self.rot_local_data_size = 200
        self.data_classes_dict = {'cifar10': 10, 'mnist': 10}
        self.data_classes = self.data_classes_dict[self.dataset_name]
        # 每个本地客户端分配的数据标签类别占比
        self.local_data_classes = 0.4
        # 数据分布
        self.data_Dis = {}
            # {0: {'data_labels': [0,1,2,3],'label_len': 0.25},
            #              1: {'data_labels': [1,3], 'label_len': 0.5},
            #              2: {'data_labels': [4,5,6,7,8], 'label_len': 0.2},
            #              3: {'data_labels': [9], 'label_len': 1},
            #              4: {'data_labels': [4,5,6,9], 'label_len':0.25}
            #              }
        self.cluster_number = 5
        self.cuda = True
        self.save_path = ""
        if len(self.data_Dis) > 0:
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
        self.DataLen = 0
        # 描述当前客户端最新的参与训练的轮次
        self.TrainRound = Round

    def set_client_info(self, ModelDict, DataSize):
        self.ModelStaticDict = copy.deepcopy(ModelDict)
        self.DataLen = DataSize


    def set_client_InClusterID(self, ClusterID):
        self.InClusterID = ClusterID


