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

a = torch.rand(10)

print(a>0.1)