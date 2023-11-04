import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# input is of size N x C = 3 x 5

inputs = torch.randn(1, 3)
print(inputs)
 # each element in target has to have 0 <= value < C
target = torch.tensor([[1.,0.2,3.]])
output = F.kl_div(F.relu(inputs), target)
output.backward()