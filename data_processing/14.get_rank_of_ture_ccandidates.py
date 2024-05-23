import numpy as np
import torch

rel_values = torch.tensor([0.6, 0.2, 0.3, 0.4, 0.5])

_, argsort1 = torch.sort(rel_values, descending=False)
argsort1 = argsort1.cpu().numpy()
#

for i in range(3):

    rank1 = np.where(argsort1 == i)[0][0] #
    print(rank1)

