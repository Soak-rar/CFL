import torch
import Model
import numpy as np
import copy
import Args
import HCFedAvg.FileProcess as FP

model = Model.init_model('mnist')

pre_dict = copy.deepcopy(model.state_dict())

quant_dict = model.quant_test()
dequant_dict = model.dequant()

scale, zero_point = model.Quanter.layers_scale_zero_point['fc4.bias']
# res = scale * (q - zero_point)
res_list = []
q_list = []
for value in model.state_dict()['fc4.bias']:
    q = value.item()
    q = round(q / scale + zero_point)
    q_list.append(q)
    res = scale * (q - zero_point)
    res_list.append(res)
print(q_list)
print(quant_dict['fc4.bias'])
print(res_list)
print(model.state_dict()['fc4.bias'])
print(dequant_dict['fc4.bias'])
# for name, param in model.state_dict().items():
#     print(name)
#     print(param - dequant_dict[name])
