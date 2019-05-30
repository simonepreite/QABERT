import torch
from torch import nn
import math
from GELU import GELU

class FeedForward(nn.Module):
	def __init__(self, hiddenSize, innerLayerDimension, dropOutProb=0.1):
		super(FeedForward, self).__init__()
		self.activationFuncion = GELU()
		self.dropout = nn.Dropout(dropOutProb)
		self.w1 = nn.Linear(hiddenSize, innerLayerDimension)
		self.w2 = nn.Linear(innerLayerDimension, hiddenSize)
		
	def forward(self, tensor):
		intermediate = self.activationFuncion(self.w1(tensor))
		linearOut = self.w2(intermediate)
		return self.dropout(linearOut)