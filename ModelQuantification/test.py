import torch
import Model
import numpy as np
import copy
import Args
import HCFedAvg.FileProcess as FP


init_model = Model.init_model('mnist')

quanter = Model.SpareBinaryQuanter()
quanter.set_spare_rate(0.1)

init_model.set_quanter(quanter)

quant_dict = init_model.quant()
for name, param in quant_dict.items():
    print(name)
    print(param)
# print(copy_dict)
# for name, param in init_model.named_parameters():
#     print(param.dtype)
