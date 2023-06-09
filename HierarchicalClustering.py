import numpy as np

import Args
import Data
from ClusterTree import BinaryTree
import pandas as pd

# 计算两个 簇 间的 平均距离
# def average_distance(BinaryNode1: BinaryTree, BinaryNode2: BinaryTree, distance_dict: {}):
#     distance_sum = 0
#     for point_1 in BinaryNode1.getRootVal():
#         for point_2 in BinaryNode2.getRootVal():
#             distance_sum += distance_dict[point_1][point_2]**2
#
#     distance_sum /= (len(BinaryNode1.getRootVal())+len(BinaryNode2.getRootVal()))
#
#     return distance_sum**0.5


# 计算两个 簇 间的 最大距离
# def average_distance(BinaryNode1: BinaryTree, BinaryNode2: BinaryTree, distance_dict: {}):
#     max_dis = 0
#     for point_1 in BinaryNode1.getRootVal():
#         for point_2 in BinaryNode2.getRootVal():
#             if max_dis < distance_dict[point_1][point_2]:
#                 max_dis = distance_dict[point_1][point_2]
#
#     return max_dis

# 计算两个 簇 间的 最小距离
# def average_distance(BinaryNode1: BinaryTree, BinaryNode2: BinaryTree, distance_dict: {}):
#     min_dis = 0
#     for point_1 in BinaryNode1.getRootVal():
#         for point_2 in BinaryNode2.getRootVal():
#             if min_dis > distance_dict[point_1][point_2]:
#                 min_dis = distance_dict[point_1][point_2]
#
#     return min_dis


# ward
# def average_distance(BinaryNode1: BinaryTree, BinaryNode2: BinaryTree, distance_dict: {}):
#     min_dis = 2
#     dict_id_pair = [0, 0]
#     for i, point_dict in distance_dict.items():
#         for j, dis in distance_dict.items():
#             if min_dis > dis:
#                 min_dis = dis
#                 dict_id_pair = [i, j]


#
# def min_cluster_distance(forest_cluster: [], distance_dict: {}}):
#     min_distance = 2
#     pair_index = [0, 0]
#     # print(len(forest_cluster))
#     for i in range(len(forest_cluster)-1):
#         for j in range(i+1, len(forest_cluster)):
#             # print(j)
#             dis = average_distance(forest_cluster[i], forest_cluster[j], distance_dict)
#             if dis < min_distance:
#                 min_distance = dis
#                 pair_index = [i, j]
#     node1 = forest_cluster.pop(pair_index[0])
#     node2 = forest_cluster.pop(pair_index[1])
#
#     val = node1.getRootVal() + node2.getRootVal()
#
#     newNode = BinaryTree(val)
#
#     newNode.leftChild = node1
#     newNode.rightChild = node2
#
#     forest_cluster.append(newNode)
#     for i in forest_cluster:
#         print(i.getRootVal())
#     print("--------------------------")
def min_cluster_distance(forest_clusters: {}, distance_dict: {}):
    min_dis = 2
    dict_id_pair = [0, 0]
    for i, point_dict in distance_dict.items():
        for j, dis in distance_dict.items():
            if min_dis > dis:
                min_dis = dis
                dict_id_pair = [i, j]

    node1 = distance_dict.pop(dict_id_pair[0])
    node2 = distance_dict.pop(dict_id_pair[1])

    new_node_ID = max(distance_dict.keys())+1

    for point_dict in distance_dict.values():
        point_dict.pop(dict_id_pair[0])
        point_dict.pop(dict_id_pair[1])

    new_node_dict = {}

    for i, point_dict in distance_dict.items():
        distance_dict[i][new_node_ID] = node1[i]


def main(load_path, train_workers):
    # 加载 节点的 距离 字典 {nodeId_i: {nodeId_j: distance, ...}, ...}
    distance_dict = np.load(load_path, allow_pickle=True).tolist()
    # 自底向上，循环开始，每个节点 作为根节点，管理森林
    # 初始化 森林簇
    forest_clusters = {i: [i] for i in distance_dict.keys()}

    while len(forest_clusters) != 4:
        min_cluster_distance(forest_clusters, distance_dict)

    data_frame = [{"簇" + str(i + 1): 0 for i in range(4)} for j in range(10)]

    for i, node in enumerate(forest_clusters):
        labels = {key: 0 for key in range(10)}
        for clients_id in node.getRootVal():
            train_worker = train_workers[clients_id]
            for key, value in train_worker["data_len"].items():
                labels[key] += value
                data_frame[key]["簇" + str(i + 1)] += value

        print("簇： {} \n 标签： {}".format(i, labels))
    pd.DataFrame(data_frame).to_csv("HierarchicalClustering/Hierarchical_labels_distribution_min.csv")


if __name__ == '__main__':
    loadPath = "HierarchicalClustering/dis_matrix_1.npy"
    args = Args.Arguments()
    train_workers = Data.load_data(args)
    main(loadPath, train_workers)



