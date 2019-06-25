#!/usr/bin/env python3

import torch
import torch.nn as nn
from NormLayer import NormLayer
import math


class BERTEmbeddings(nn.Module):

	def __init__(self, hiddenSize, vocabSize, maxPosEmbedding, dropoutProb=0.1):
		super(BERTEmbeddings, self).__init__()
		self.maxLen = maxPosEmbedding
		self.wordEmbeddings = nn.Embedding(vocabSize, hiddenSize, padding_idx=0)
		self.posEmbeddings = nn.Embedding(self.maxLen, hiddenSize)
		self.seqEmbeddings = nn.Embedding(2, hiddenSize)
		self.normLayer = NormLayer(hiddenSize)
		self.dropout = nn.Dropout(dropoutProb)

	def forward(self, inputIDs, sequenceIDs):
		we = self.wordEmbeddings(inputIDs)
		se = self.seqEmbeddings(sequenceIDs)
		
		seqLength = inputIDs.size(1)
		posIDs = torch.arange(seqLength, dtype=torch.long, device=inputIDs.device)
		posIDs = posIDs.unsqueeze(0).expand_as(inputIDs)
		pe = self.posEmbeddings(posIDs)

		embeddings = self.normLayer(we + pe + se)
		return self.dropout(embeddings)

		
