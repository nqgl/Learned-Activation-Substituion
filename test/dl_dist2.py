import transformer_lens
import circuitsvis as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from jaxtyping import Float
import transformer_lens.utils as utils
# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("distilgpt2")
model2 = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
