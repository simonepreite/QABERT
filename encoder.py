import torch
from torch import nn

class ScaleDotProduct(nn.Module):
    def __init__(self, query, keys, values):
        self.Q = query
        self.K = keys
        self.V = values
        
    def forward(self, hiddenStates, attentionMask, attentionHeadSize):
        aScores = torch.matmul(self.Q, self.K.transpose(-1, -2)) / math.sqrt(attentionHeadSize)
        aScores = aScores + attentionMask
        aProbs = nn.Dropout(nn.Softmax(dim=-1)(aScores))
        torch.matmul(aProbs, self.V).contiguous()
        
        
    