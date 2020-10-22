import torch
from torch.nn import functional as F
from utils import prob_to_logit

probs = F.softmax(torch.randn([2,3]))
print(prob_to_logit(probs))
