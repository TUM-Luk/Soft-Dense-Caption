import json
import os
import torch
from lib.config import CONF
import numpy as np
import torch.nn.functional as F

a = torch.zeros(10)

a[5] = 1

b=np.where(a==2)

print(b[0].size)
