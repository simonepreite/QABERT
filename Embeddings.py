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
		hiddenSize = we.size(1)
		pe = torch.zeros(self.maxLen, hiddenSize)
		pe.require_grad = False

		positionIDs = torch.arange(self.maxLen).unsqueeze(1)
		normTerm = (torch.arange(hiddenSize,2) * -(math.log(10000) / hiddenSize)).exp()

		pe[:, 0::2] = torch.sin(positionIDs * normTerm)
		pe[:, 1::2] = torch.cos(positionIDs * normTerm)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		pe = self.pe[:, inputIDs.size(1)]

		embeddings = self.normLayer(we + pe + se)
		return self.dropout(embeddings)

		