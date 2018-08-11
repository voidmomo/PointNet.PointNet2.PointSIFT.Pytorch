# coding:utf-8

import numpy as np
import torch

# a = np.random.rand(3, 2, 2, 3)
# b = np.random.rand(3, 3, 2, 3)
#
# # out = np.concatenate([a, b], axis=0)
# # print(out.shape)
# #
# # out_t = torch.from_numpy(out)
# # print(out_t)
#
# a = torch.tensor([[1, 3, 2], [3, 4, 5], [5, 6, 7]])
# b = torch.tensor([3, 1, 2])
# import torch.nn.functional as F
#
# import torch.nn as nn
#
# #
# # nn.NLLLoss
# # F.nll_loss()
#
# net = nn.Sequential(
#     nn.Linear(10, 2)
# )
#
# inp = torch.rand(10)
# outp = net(inp)
# # print(outp.size())
# #
# # print(net.state_dict())
# import torch.optim as optim
#
# opt = optim.SGD(net.parameters(), lr=1e-3)
# for param in opt.param_groups:
#     print(type(param['lr']))
# # print(tuple(inp.size()[0]) + tuple([-1,]))
# a = max(2.3, 3.4)
# print(a)
a = np.ones([3, 2, 3])
print(a)
print(a[1, ...])
b = a[1, ...]
c = np.array([[2, 2, 2], [3, 3, 3], [5, 5, 5]])
d = np.dot(b, c)
print(d)

