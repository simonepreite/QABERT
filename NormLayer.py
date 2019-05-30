#!/usr/bin/env python3

import torch
import torch.nn as nn


class NormLayer(nn.Module):
	"""
		Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450)
		It consists of Batch Normalization Transform to speed up learning with mean and std computed according to the above paper

		normWeights:
			weights for this normalization layer which will be learnt during training
		normBias:
			bias for this normalization layer which will be learnt during training
		epsilon:
			numerical stability parameter to avoid division by zero
	"""

	def __init__(self, hiddenSize, epsilon=1e-12):
		super(NormLayer, self).__init__()
		self.normWeights = nn.Parameter(torch.ones(hiddenSize))
		self.normBias = nn.Parameter(torch.zeros(hiddenSize))
		self.epsilon = epsilon


	def forward(self, input):
		mu = input.mean(-1, keepdim=True)
		stdArg = (input - mu).pow(2).mean(-1, keepdim=True) + self.epsilon
		std = torch.sqrt(stdArg)
		normInput = (input - mu) / std
		return self.normWeights * normInput + self.normBias

