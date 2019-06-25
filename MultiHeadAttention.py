#!/usr/bin/env python3

import torch
import torch.nn as nn
from ScaledDotProduct import ScaledDotProduct


class MultiHeadAttention(nn.Module):

	def __init__(self, hiddenSize, numAttentionHeads, dropoutProb):
		super(MultiHeadAttention, self).__init__()

		if hiddenSize % numAttentionHeads != 0:
			raise ValueError("The hidden size ({}) is not a multiple of the numeber of attention heads ({})".format(hiddenSize, numAttentionHeads))

		self.numAttentionHeads = numAttentionHeads
		self.attentionHeadSize = hiddenSize // self.numAttentionHeads

		self.queriesLinear = nn.Linear(hiddenSize, hiddenSize)
		self.keysLinear = nn.Linear(hiddenSize, hiddenSize)
		self.valuesLinear = nn.Linear(hiddenSize, hiddenSize)
		self.sdp = ScaledDotProduct(self.attentionHeadSize, dropoutProb)
		self.outputLinear = nn.Linear(hiddenSize, hiddenSize)

	def prepareForScores(self, input):
		newShape = input.size()[:-1] + (self.numAttentionHeads, self.attentionHeadSize)
		input = input.view(*newShape)
		return x.permute(0, 2, 1, 3)
		# batchSize = input.size(0)
		# input = input.view(batchSize, -1, self.numAttentionHeads, self.attentionHeadSize)
		# return input.transpose(1, 2)

	def forward(self, hiddenStates, attentionMask):
		projQ = self.queriesLinear(hiddenStates)
		projK = self.keysLinear(hiddenStates)
		projV = self.valuesLinear(hiddenStates)

		queries = self.prepareForScores(projQ)
		keys = self.prepareForScores(projK)
		values = self.prepareForScores(projV)

		attentionScores = self.sdp(queries, keys, values, attentionMask)

		attentionScores = attentionScores.permute(0, 2, 1, 3).contiguous()
		newShape = attentionScores.size()[:-2] + (self.numAttentionHeads * self.attentionHeadSize,)
		attentionScores = attentionScores.view(*newShape)
		# batchSize = queries.size(0)
		# attentionScores = attentionScores.transpose(1, 2).contiguous()
		# attentionScores = attentionScores.view(batchSize, -1, self.numAttentionHeads * self.attentionHeadSize)

		return self.outputLinear(attentionScores)

		


		
