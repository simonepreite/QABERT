#!/usr/bin/env python3

import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from NormLayer import NormLayer
from FeedForward import FeedForward


class Encoder(nn.Module):

	def __init__(self, hiddenSize, numAttentionHeads, normEpsilon=1e-12, dropoutProb=0.1):
		super(Encoder, self).__init__()

		self.multiHeadAtt = MultiHeadAttention(hiddenSize, numAttentionHeads, dropoutProb)
		self.feedForward = FeedForward(hiddenSize, 4 * hiddenSize, dropoutProb)
		self.attNorm = NormLayer(hiddenSize, normEpsilon)
		self.outputNorm = NormLayer(hiddenSize, normEpsilon)
		self.dropout = nn.Dropout(dropoutProb)

	def forward(self, hiddenStates, attentionMask):
		attentionOutput = self.multiHeadAtt(hiddenStates, attentionMask)

		# Add+Norm for MultiHeadAttention output
		attentionOutput = self.dropout(attentionOutput)
		normAttOutput = self.attNorm(attentionOutput + hiddenStates)

		ffOutput = self.feedForward(normAttOutput)
		normFFOutput = self.outputNorm(ffOutput + normAttOutput)

		return normFFOutput