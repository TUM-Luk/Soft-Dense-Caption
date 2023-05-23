import json
import os
import torch
from lib.config import CONF
import numpy as np
import torch.nn.functional as F


import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge


# caption_decoded= {'test': ['sos there is a black lujiachen eos']}
# gt_caption={'test': ['sos there is a black dog eos']}
#
# bleu = capblue.Bleu(4).compute_score(gt_caption,caption_decoded)
# cider = capcider.Cider().compute_score(gt_caption,caption_decoded)
# rouge = caprouge.Rouge().compute_score(gt_caption,caption_decoded)
# print(bleu)
# print(cider)
# print(rouge)
#
# print('test' in caption_decoded.keys())

a= torch.rand(4,20,10)
b = torch.zeros(4,10)
b[0][9]=1
b.unsqueeze(dim=1)

# print(b.unsqueeze(dim=1).repeat(1,20,1))
# print(b.unsqueeze(dim=1).repeat(1,20,1).shape)
print(a+b.unsqueeze(dim=1))
print(torch.cat((a,b.unsqueeze(dim=1).repeat(1,20,1)),dim=2))

