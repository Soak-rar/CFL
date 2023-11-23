import math
from typing import Dict, List
import copy

import numpy as np

from Args import ClientInServerData

import Model


class HCCluster:
    def __init__(self, init_clients: List[int], similarity_matrix: Dict[int, Dict[int, float]]):
        self.Clients: List[int] = init_clients
        self.MaxInClusterDistance: float = 0.0
        self.ClientNumber = len(init_clients)
        self.set_max_in_cluster_distance(similarity_matrix)
        self.AvgClusterModelDict = None
        self.ClusterResDictUpdate = None
        self.CurrentModelRound = 0


    def get_avg_cluster_model_copy(self):
        return copy.deepcopy(self.AvgClusterModelDict)

    def get_avg_cluster_res_copy(self):
        return copy.deepcopy(self.ClusterResDictUpdate)

    def set_cluster_res_update(self, clients_model: Dict[int, ClientInServerData], train_clients):
        if len(train_clients) == 0:
            return
        self.ClusterResDictUpdate = copy.deepcopy(clients_model[train_clients[0]].ModelStaticDict)

        # 找到 最新的 全局残差，并进行聚合

        max_res_round = 0
        max_res_clients = []

        total_len = 0
        if len(train_clients) > 1:
            for client_id in train_clients:
                total_len += clients_model[client_id].DataLen
                if clients_model[client_id].LocalToGlobalResRound > 0:
                    if clients_model[client_id].LocalToGlobalResRound > max_res_round:
                        max_res_clients.clear()
                        max_res_clients.append(client_id)
                        max_res_round = clients_model[client_id].LocalToGlobalResRound
                    elif clients_model[client_id].LocalToGlobalResRound == max_res_round:
                        max_res_clients.append(client_id)

            for key in self.ClusterResDictUpdate.keys():
                self.ClusterResDictUpdate[key] *= 0
                for client_id in max_res_clients:
                    self.ClusterResDictUpdate[key] += (clients_model[client_id].LocalToGlobalResDictUpdate[key] * clients_model[
                        client_id].DataLen / total_len)


    # 计算当前集群的距离标准差
    def calculate_sd(self, similarity_matrix: Dict[int, Dict[int, float]]):
        if len(self.Clients) == 1:
            return 0.0, 0.0

        DisSum = 0.0
        for client_id_l in self.Clients:
            for client_id_r in self.Clients:
                if client_id_r != client_id_l:
                    DisSum += similarity_matrix[client_id_l][client_id_r]

        DisAvg = DisSum / (len(self.Clients) * (len(self.Clients) - 1))

        DifferSum = 0.0
        for client_id_l in self.Clients:
            for client_id_r in self.Clients:
                if client_id_r != client_id_l:
                    DifferSum += (similarity_matrix[client_id_l][client_id_r] - DisAvg)**2

        DifferAvg = DifferSum / (len(self.Clients) * (len(self.Clients) - 1))
        return DifferAvg ** 0.5, DisAvg

    def set_avg_cluster_model(self, clients_model: Dict[int, ClientInServerData], train_clients):
        if len(train_clients) == 0:
            return
        self.AvgClusterModelDict = copy.deepcopy(clients_model[train_clients[0]].ModelStaticDict)
        MaxRound = 0
        total_len = 0
        if len(train_clients) > 1:
            for client_id in train_clients:
                total_len += clients_model[client_id].DataLen
                if clients_model[client_id].TrainRound > MaxRound:
                    MaxRound = clients_model[client_id].TrainRound

            self.CurrentModelRound = MaxRound

            for key in self.AvgClusterModelDict.keys():
                self.AvgClusterModelDict[key] *= 0
                for client_id in train_clients:
                    self.AvgClusterModelDict[key] += (clients_model[client_id].ModelStaticDict[key] * clients_model[client_id].DataLen / total_len)



    def set_avg_cluster_model_with_time(self, clients_model: Dict[int, ClientInServerData], train_clients):

        # self.AvgClusterModelDict = copy.deepcopy(clients_model[self.Clients[0]].ModelStaticDict)
        # MaxRound = 0
        # e_sum = 0.0
        # for client_id in train_clients:
        #     client_round = clients_model[client_id].TrainRound
        #     e_sum += math.exp(client_round)
        #     if clients_model[client_id].TrainRound > MaxRound:
        #         MaxRound = clients_model[client_id].TrainRound
        #
        # self.CurrentModelRound = MaxRound
        #
        # for key in self.AvgClusterModelDict.keys():
        #     self.AvgClusterModelDict[key] *= 0
        #     for client_id in train_clients:
        #         self.AvgClusterModelDict[key] += (clients_model[client_id].ModelStaticDict[key]) * (math.exp(clients_model[client_id].TrainRound)) / e_sum

        self.AvgClusterModelDict = copy.deepcopy(clients_model[self.Clients[0]].ModelStaticDict)
        MaxRound = 0
        e_sum = 0.0
        total_len = 0
        for client_id in train_clients:
            client_round = clients_model[client_id].TrainRound
            total_len += clients_model[client_id].DataLen
            if clients_model[client_id].TrainRound > MaxRound:
                MaxRound = clients_model[client_id].TrainRound

        self.CurrentModelRound = MaxRound

        for key in self.AvgClusterModelDict.keys():
            self.AvgClusterModelDict[key] *= 0
            for client_id in train_clients:
                self.AvgClusterModelDict[key] += (clients_model[client_id].ModelStaticDict[key]) *  clients_model[client_id].DataLen / total_len

    def set_max_in_cluster_distance(self, similarity_matrix: Dict[int, Dict[int, float]]):
        for client_id_l in self.Clients:
            for client_id_r in self.Clients:
                if self.MaxInClusterDistance < similarity_matrix[client_id_l][client_id_r]:
                    self.MaxInClusterDistance = similarity_matrix[client_id_l][client_id_r]

    def get_max_in_cluster_distance(self) -> float:
        return self.MaxInClusterDistance

    def get_clients_list(self) -> List[int]:
        return self.Clients

# 管理 当前的所有 集群
class HCClusterManager:
    def __init__(self):
        self.CurrentClusters: Dict[int, HCCluster] = {}
        self.CurrentSimilarityMatrix: Dict[int, Dict[int, float]] = {}
        self.ClusterSimilarityMatrix: Dict[int, Dict[int, float]] = {}
        self.H = 0.06
        # self.H = 0.2
        # init_clusters(self, client_number)

    def get_cluster_by_id(self, ClusterID):
        return self.CurrentClusters[ClusterID]

    def reset_similarity_matrix(self, similarity_matrix):
        self.CurrentSimilarityMatrix = copy.deepcopy(similarity_matrix)
        self.ClusterSimilarityMatrix = copy.deepcopy(similarity_matrix)
        for client_id, Client_sim in similarity_matrix.items():
            self.CurrentClusters[client_id] = HCCluster(list([client_id]), self.CurrentSimilarityMatrix)

    # 循环集群划分
    def HCClusterDivide(self):
        while True and len(self.CurrentSimilarityMatrix) > 1 and len(self.CurrentClusters) > 1:
            TempClusters = copy.deepcopy(self.CurrentClusters)
            # 找到集群相似度矩阵 中的最小距离值
            MinDisBetweenTwoClusters = 99
            MinClusterIDPair = [0, 0]
            for Cluster_ID, OtherClustersList in self.ClusterSimilarityMatrix.items():
                for OtherCluster_ID, OtherCluster_Dis in OtherClustersList.items():
                    if MinDisBetweenTwoClusters > OtherCluster_Dis and Cluster_ID != OtherCluster_ID:
                        MinDisBetweenTwoClusters = OtherCluster_Dis
                        MinClusterIDPair[0] = Cluster_ID
                        MinClusterIDPair[1] = OtherCluster_ID

            if MinDisBetweenTwoClusters > self.H:
                print('  迭代-----------')
                for cluster_id, cluster in self.CurrentClusters.items():
                    print(cluster_id, "  , ", cluster.Clients)
                # self.calculate_clusters_sd()
                break

            IsMerge  = False
            if TempClusters[MinClusterIDPair[0]].ClientNumber == 1 and TempClusters[MinClusterIDPair[1]].ClientNumber == 1:
                # 直接合并
                IsMerge = True
            else: # 计算是否合并
                # 获取两个集群中 簇内最大距离
                MaxInClusterDis = max(TempClusters[MinClusterIDPair[0]].get_max_in_cluster_distance(), TempClusters[MinClusterIDPair[1]].get_max_in_cluster_distance())

                # 获取两个集群中 簇间最小距离
                MinOutClusterDis = calculate_cluster_min_distance(TempClusters[MinClusterIDPair[0]], TempClusters[MinClusterIDPair[1]], self.CurrentSimilarityMatrix)

                # 判断 簇内最大距离和 簇间最小距离
                if MinOutClusterDis <= MaxInClusterDis:
                    IsMerge = True

            # if IsMerge is False:
            #     break
            # else:
            if True:
                MergeClientList = TempClusters[MinClusterIDPair[0]].get_clients_list()[:]
                MergeClientList.extend(TempClusters[MinClusterIDPair[1]].get_clients_list())
                TempClusters[MinClusterIDPair[0]] = HCCluster(MergeClientList, self.CurrentSimilarityMatrix)
                TempClusters.pop(MinClusterIDPair[1])

                # 更新集群相似度矩阵
                self.ClusterSimilarityMatrix.pop(MinClusterIDPair[1])
                for key, value in self.ClusterSimilarityMatrix.items():
                    value.pop(MinClusterIDPair[1])

                for key, value in TempClusters.items():
                    if key != MinClusterIDPair[0]:
                        new_dis = calculate_cluster_avg_distance(TempClusters[MinClusterIDPair[0]], value,
                                                                 self.CurrentSimilarityMatrix)

                        # new_dis = calculate_cluster_min_dis(TempClusters[MinClusterIDPair[0]], value,
                        #                                          self.CurrentSimilarityMatrix)
                        #
                        # new_dis = calculate_cluster_max_dis(TempClusters[MinClusterIDPair[0]], value,
                        #                                          self.CurrentSimilarityMatrix)

                        self.ClusterSimilarityMatrix[MinClusterIDPair[0]][key] = new_dis
                        self.ClusterSimilarityMatrix[key][MinClusterIDPair[0]] = new_dis
                self.CurrentClusters = copy.deepcopy(TempClusters)


    def UpdateClusterAvgModel(self, clients_model: Dict[int, ClientInServerData], cluster_clients_train:[int]):
        for cluster_id, Cluster in self.CurrentClusters.items():
            for client_id in Cluster.Clients:
                clients_model[client_id].set_client_InClusterID(cluster_id)
            train_clients = self.get_last_train_clients(clients_model, Cluster.Clients)
            Cluster.set_avg_cluster_model(clients_model, train_clients)

    # 设置集群模型 ，如果全局轮次满足条件设置集群残差
    def UpdateClusterAvgModelAndResWithTime(self, clients_model: Dict[int, ClientInServerData], use_quant = True):
        for cluster_id, Cluster in self.CurrentClusters.items():
            for client_id in Cluster.Clients:
                clients_model[client_id].set_client_InClusterID(cluster_id)
            train_clients = self.get_last_train_clients(clients_model, Cluster.Clients)
            Cluster.set_avg_cluster_model_with_time(clients_model,train_clients)
            if use_quant:
                Cluster.set_cluster_res_update(clients_model, Cluster.Clients)

    def get_last_train_clients(self, clients_model: Dict[int, ClientInServerData], Clients:[int]):
        max_round = 0
        train_clients = []
        for client_id in Clients:
            if clients_model[client_id].TrainRound > max_round:
                train_clients.clear()
                train_clients.append(client_id)
                max_round = clients_model[client_id].TrainRound
            elif clients_model[client_id].TrainRound == max_round:
                train_clients.append(client_id)
        return train_clients

    # def print_divide_result(self):
    #     print("     划分结果     ")
    #     for cluster_id, Cluster in self.CurrentClusters.items():
    #         print("集群： {},  客户端：{}".format(cluster_id, Cluster.Clients))

    def calculate_clusters_sd(self):
        dict_clients = {}
        avg_list = []
        std_list = []

        for cluster_id, ClusterClass in self.CurrentClusters.items():
            for client_id in ClusterClass.Clients:
                dict_clients[client_id] = 0

        print('当前的集群数量： ', len(self.CurrentClusters))
        print('当前的参与过训练的客户端数量： ', len(dict_clients))
        # print("    集群标准差    ")
        for cluster_id, ClusterClass in self.CurrentClusters.items():

            sd_value, avg_value = ClusterClass.calculate_sd(self.CurrentSimilarityMatrix)
            avg_list.append(avg_value)
            std_list.append(sd_value)
            # print("集群： {}, 标准差： {}, 平均数: {}".format(cluster_id, sd_value, avg_value))

        return np.mean(avg_list), np.mean(std_list)

def init_clusters(HCManager: HCClusterManager, client_number: int):
    for i in range(client_number):
        HCManager.CurrentClusters[i] = HCCluster(list([i]), HCManager.CurrentSimilarityMatrix)

# 计算集群间的相似性，由集群A中 和 集群B的 距离最近的 两个点 表示
def calculate_cluster_min_distance(Cluster_A: HCCluster, Cluster_B: HCCluster, similarity_matrix: Dict[int, Dict[int, float]]) -> float:
    MinOutClusterDis = 2
    for Client_A in Cluster_A.get_clients_list():
        for Client_B in Cluster_B.get_clients_list():
            if similarity_matrix[Client_A][Client_B] < MinOutClusterDis:
                MinOutClusterDis = similarity_matrix[Client_A][Client_B]
    return MinOutClusterDis


# 计算集群间的 平均相似度
def calculate_cluster_avg_distance(Cluster_A: HCCluster, Cluster_B: HCCluster, similarity_matrix: Dict[int, Dict[int, float]]) -> float:
    DisSum = 0.0
    for Client_A in Cluster_A.get_clients_list():
        for Client_B in Cluster_B.get_clients_list():
            DisSum += similarity_matrix[Client_A][Client_B]

    return DisSum / (Cluster_A.ClientNumber * Cluster_B.ClientNumber)


def calculate_cluster_min_dis(Cluster_A: HCCluster, Cluster_B: HCCluster, similarity_matrix: Dict[int, Dict[int, float]]) -> float:
    DisMin = 2.0
    for Client_A in Cluster_A.get_clients_list():
        for Client_B in Cluster_B.get_clients_list():
            if similarity_matrix[Client_A][Client_B] < DisMin:
                DisMin = similarity_matrix[Client_A][Client_B]

    return DisMin


def calculate_cluster_max_dis(Cluster_A: HCCluster, Cluster_B: HCCluster, similarity_matrix: Dict[int, Dict[int, float]]) -> float:
    DisMax = 0.0
    for Client_A in Cluster_A.get_clients_list():
        for Client_B in Cluster_B.get_clients_list():
            if similarity_matrix[Client_A][Client_B] > DisMax:
                DisMax = similarity_matrix[Client_A][Client_B]

    return DisMax