import torch
from torch import nn

class BERT(nn.Module):
	def __init__(self, hiddenSize=768, numLayers=12, numAttentionHeads=12,vocabSize, dropout=0.1):
		super(BERT, self).__init__()
		self.hiddenSize = hiddenSize
		self.numLayers = numLayers
		self.numAttentionHeads = numAttentionHeads
		self.maxPosEmbedding = 512
		
		self.embedding = BERTEmbeddings(self.hiddenSize, self.vocabSize, self.maxPosEmbedding)
		self.encoder = nn.Sequential([Encoder(hiddenSize, numAttentionHeads) for _ in range(numLayers)])
		
		
	def forward(tensor, sequenceIDs):
		x = torch.FloatTensor([[[ 0.9059, -0.7039, -0.3376,  0.1968],[-1.0413,  0.8128,  0.0697, -0.6166],[-0.3793, -0.9851, -2.3841, -0.7003],[ 0.6076, -1.4874, -0.1079,  0.4266]]])

		inputIDs = torch.randn(1, 4)
		attentionMask = torch.ones_like(inputIDs)
		extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2)
		#extendedAttentionMask = extendedAttentionMask.to(dtype=next(self.parameters()).dtype)
		extendedAttentionMask = (1.0 - extendedAttentionMask) * -10000.0
		
		x = self.embedding(x, sequenceIDs)
		x = self.encoder(x, extendedAttentionMask)
		
		