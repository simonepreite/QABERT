import torch
from torch import nn
import math

class ScaledDotProduct(nn.Module):
    def __init__(self, attentionHeadSize, dropOutProb=0.1):
        super().__init__()
        self.attentionHeadSize = attentionHeadSize
        self.dropout = nn.Dropout(dropOutProb)
        
    def forward(self, Q, K, V, attentionMask):
        aScores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attentionHeadSize)
        aScores = aScores + attentionMask
        aProbs = self.dropout(nn.Softmax(dim=-1)(aScores))
        return torch.matmul(aProbs, V), aProbs