#!/usr/bin/env python3

import torch
import torch.nn as nn
from .ScaledDotProduct


class MultiHeadAttention(nn.Module):

	def __init__(self, hiddenSize, numAttentionHeads, dropoutProb):
		super(MultiHeadAttention, self).__init__()

		self.numAttentionHeads = numAttentionHeads
		self.attentionHeadSize = hiddenSize // self.numAttentionHeads

		self.queriesLinear = nn.Linear(hiddenSize, hiddenSize)
		self.keysLinear = nn.Linear(hiddenSize, hiddenSize)
		self.valuesLinear = nn.Linear(hiddenSize, hiddenSize)
		self.sdp = ScaledDotProduct(self.attentionHeadSize, dropoutProb)
		self.outputLinear = nn.Linear(hiddenSize, hiddenSize)

		self.dropout = nn.Dropout(dropoutProb)

	def prepareForScores(self, input):
		batchSize = input.size(0)
		input = input.view(batchSize, -1, self.numAttentionHeads, self.attentionHeadSize)
		return input.transpose(1, 2)

	def forward(self, hiddenStates, attentionMask):
		projQ = self.queriesLinear(hiddenStates)
		projK = self.keysLinear(hiddenStates)
		projV = self.valuesLinear(hiddenStates)

		queries = self.prepareForScores(projQ)
		keys = self.prepareForScores(projK)
		values = self.prepareForScores(projV)

		attentionScores, _ = self.sdp(queries, keys, values, attentionMask)

		batchSize = queries.size(0)
		attentionScores = attentionScores.transpose(1, 2).contiguous()
		attentionScores = attentionScores.view(batchSize, -1, hiddenSize)

		return self.outputLinear(attentionScores)

		


		
