import multiprocessing
import time

from DataScientist import pca_dim_deduction


def test_list(l):
    for l_ in l:
        l_.remove(1)


def newProcess(new_dict):
    print(new_dict[0])


import matplotlib.pyplot as plt
import numpy as np

def draw_dynamic():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title("10-Class Data Distribution")
    # if len(data_) > 0:
    #     for key, value in clients_dict:
    #         data_[key] = pca_dim_deduction(value[0], 3)
    #     point_style_list = ['lightcoral', 'darkkhaki', 'green', 'lightblue', 'mistyrose']
    #     for id, workers_id in enumerate(clients_clusters):
    #         for worker_id in workers_id:
    i = 1
    while True:
        print("画")
        i += 1
        ax.scatter(i, i, i, c="green", marker="^")


if __name__ == '__main__':
    shared_dict = multiprocessing.Manager().list()
    shared_dict.append([[1], [2], [3]])

    litst1 = [1, 2]
    list2 = [3,4]
    list4 = []
    list4.append(litst1)
    list4.append(list2)
    print(list4)


