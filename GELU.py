import torch
from torch import nn
import math

class GELU(nn.Module):
	def __init__(self):
		super(GELU, self).__init__()
		
	def forward(self, tensor):
		geluPow = tensor + 0.044715 * torch.pow(tensor, 3)
		geluTanh = torch.tanh(math.sqrt(2 / math.pi) * geluPow)
		geluResult = 1 + geluTanh
		return 0.5 * tensor * geluResult