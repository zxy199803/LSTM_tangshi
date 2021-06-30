import torch
import numpy as np

a = np.array([1,2,3])
b = torch.from_numpy(a)
print(b.topk(1))