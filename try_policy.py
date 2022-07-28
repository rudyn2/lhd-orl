import torch
from d3rlpy.algos import CQL

# load greedy-policy only with PyTorch
# policy = torch.jit.load('/home/rudy/Documents/lhd-orl/d3rlpy_logs/CQL_20220728142935/model_1878.pt')

model = CQL()
model.load_model("/home/rudy/Documents/lhd-orl/d3rlpy_logs/CQL_20220728142935/model_1878.pt")
