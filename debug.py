import numpy as np
import torch
import Args
import Data
import Model
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import os
import time


def fun(i, model):
    time.sleep(1)
    if i == 5:
        print(2 ^ 10)
    print(os.getpid())
    print(model)


def main():
    pool = mp.Pool(4)
    model = Model.AlexNet()
    model_shared = Manager().dict({'model': torch.tensor([1, 2, 3])})
    for i in range(2):
        for k in range(5):
            pool.apply_async(func=fun, args=(k, model_shared))
        print('主进程')
    pool.close()
    pool.join()
    del model_shared


if __name__ == '__main__':
    main()

# args = Args.Arguments()
# print(args.data_classes)
# model = Model.AlexNet()
# print(model)
