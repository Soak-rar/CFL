import torch
from torchsummary import summary
import Model

# 定义模型'NewCifar10'
model = Model.init_model('NewCifar10')

# 打印模型输出张量维度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, (3, 32, 32))