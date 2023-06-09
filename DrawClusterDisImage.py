import numpy as np
import torch
import collections
from KMeansPP import get_cos_dis_single_layer
from ClusterMain import pca_dim_deduction
from matplotlib import pyplot as plt
import math

def L2_Distance(tensor1, tensor2):
    # Value = 0
    # for i in range(tensor1.shape[0]):
    #     Value += math.pow(tensor1[i].item() - tensor2[i].item(), 2)
    # return Value
    #
    # rest = torch.cosine_similarity(tensor1, tensor2, dim=-1).item()
    # return 1 - rest
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

    return 1 - UpSum / (DownSum1 * DownSum2)


if __name__ == '__main__':

    Grad_Random = torch.load('DeepModelSimality/Random_Epoch_5_Grad_10_WithTestAll_LossIf_Fc4Grad_With_weightGrad_With_Param_4_Param.pt')

    for key, Key_Value in Grad_Random.items():
        for Key, Value in Key_Value.items():
            Avg_ = torch.zeros(20)
            Avg_range = [0, 10]
            for i in range(Avg_range[0], Avg_range[1]):
                Avg_ = Avg_ + Value[i]
            Avg_ = Avg_ / (Avg_range[1] - Avg_range[0])

            if Key == 0:
                print(key)

            Grad_Random[key][Key] = Avg_

    # for key, Key_Value in Grad_Random.items():
    #     Last_Key = -1
    #     print(" Client  ")
    #     for Key, Value in Key_Value.items():
    #         Avg_ = torch.zeros(20)
    #
    #         if Last_Key == -1:
    #             Grad_Random[key][Key] = torch.mean(Value, dim=0)
    #         else:
    #             Dis_dict = collections.OrderedDict({i: L2_Distance(Grad_Random[key][Last_Key][i], Grad_Random[key][Key][i]) for i in range(len(Value))})
    #             Sorted_dict = collections.OrderedDict(sorted(Dis_dict.items(), key=lambda x: x[1]))
    #             print(Sorted_dict)
    #             Sorted_Key_List = list(Sorted_dict.keys())
    #             MaxDifferIndex = 0
    #             MaxDifferValue = -99
    #             for i in range(len(Sorted_Key_List)-1):
    #                 NewDiffer = Sorted_dict[Sorted_Key_List[i+1]] - Sorted_dict[Sorted_Key_List[i]]
    #                 if NewDiffer > MaxDifferValue:
    #                     MaxDifferValue = NewDiffer
    #                     MaxDifferIndex = i + 1
    #             print(Sorted_Key_List[:MaxDifferIndex])
    #             for value_index in Sorted_Key_List[:4]:
    #                 Avg_ = Avg_ + Value[value_index]
    #             Avg_ = Avg_ / 4
    #
    #             Grad_Random[key][Key] = Avg_
    #         print(Last_Key)
    #         Last_Key = Key

    CurrentClientID = 3

    CurrentClientRounds = list(Grad_Random[CurrentClientID].keys())

    Clusters_Grad = {i: {j: [] for j in range(50)} for i in range(5)}  # 每个集群的每个轮次 的梯度列表

    fig = plt.figure(figsize=(48, 24))

    for Key, Key_Value in Grad_Random.items():
        for key, Value in Key_Value.items():
            if Key is not CurrentClientID:
                Clusters_Grad[Key % 5][key].append(Value)

    for CurrentRound, CurrentClientRound in enumerate(CurrentClientRounds):
        Lines = [{} for i in range(5)]
        for Key, Key_Value in Clusters_Grad.items():
            for key, Value in Key_Value.items():
                if len(Value) > 0:
                    total_dis = 0
                    for grad in Value:
                        total_dis += L2_Distance(Grad_Random[CurrentClientID][CurrentClientRound], grad)

                    Lines[Key][key] = total_dis / len(Value)

        ax1 = fig.add_subplot(int(len(CurrentClientRounds) / 4) + 1, 4, CurrentRound + 1)

        for cluster_id, line in enumerate(Lines):
            print(line.values())
            ax1.plot(line.keys(), line.values(), label="Cluster " + str(cluster_id + 1) + " Distance")
        # ax1.plot(np.arange(len(global_1_3['acc'])), global_1_3['acc'], label="10-Class Data Distribution")
        ax1.legend(loc='best')
        ax1.set_xlabel('Train Round')
        ax1.set_ylabel('Distance')
        ax1.set_title("Clusters Client_{}_Round_{}_InCluster_{} Similarity".format(CurrentClientID, CurrentClientRound,
                                                                                   CurrentClientID % 5 + 1))
    plt.show()
