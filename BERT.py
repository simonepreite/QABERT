import torch
from torch import nn
from NormLayer import NormLayer
from utils import loadModuleParameters
from Embeddings import BERTEmbeddings
from Encoder import Encoder
from convertWeights import convertTensorFlowWeights


class BERTInitializer(nn.Module):
	def __init__(self, *inputs, **kwargs):
		super(BERTInitializer, self).__init__()

	def weightsInitialization(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# implement truncated normal for weights init
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, NormLayer):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	@classmethod
	def loadPretrained(className, checkpointPath, usingTensorFlow, outputPath, *inputs, **kwargs):
		stateDict = kwargs.get("state_dict", None)
		kwargs.pop("state_dict", None)

		model = className(*inputs, **kwargs)

		if stateDict is None:
			if usingTensorFlow:
				convertTensorFlowWeights(model, checkpointPath, outputPath)
				return model
			else:
				stateDict = torch.load(checkpointPath)

		model.load_state_dict(stateDict)

		return model


class BERTModel(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(BERTModel, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.hiddenSize = hiddenSize
		self.numLayers = numLayers
		self.numAttentionHeads = numAttentionHeads
		self.maxPosEmbedding = 512
		self.vocabSize = vocabSize

		self.apply(self.weightsInitialization)
		
		self.embeddings = BERTEmbeddings(self.hiddenSize, self.vocabSize, self.maxPosEmbedding)
		self.encoder = nn.ModuleList(Encoder(hiddenSize, numAttentionHeads) for _ in range(numLayers))
		
		
	def forward(self, inputIDs, sequenceIDs, attentionMask=None):
		print("BERTModel forward")
		#x = torch.FloatTensor([[[ 0.9059, -0.7039, -0.3376,  0.1968],[-1.0413,  0.8128,  0.0697, -0.6166],[-0.3793, -0.9851, -2.3841, -0.7003],[ 0.6076, -1.4874, -0.1079,  0.4266]]])

		#inputIDs = torch.randn(1, 4)
		if attentionMask is None:
			attentionMask = torch.ones_like(inputIDs)
		extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2)
		extendedAttentionMask = extendedAttentionMask.to(dtype=next(self.parameters()).dtype)
		extendedAttentionMask = (1.0 - extendedAttentionMask) * -10000.0
		
		embeddingOutput = self.embeddings(inputIDs, sequenceIDs)
		print("embedding output shape:", embeddingOutput.size())
		encodedLayers = embeddingOutput
		for layer in self.encoder:
			encodedLayers = layer(encodedLayers, extendedAttentionMask)
		print("last encoded layer shape:", encodedLayers.size())
		return encodedLayers
		

class QABERT(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.apply(self.weightsInitialization)
		self.midIsImpossibleLinear = nn.Linear(hiddenSize, 256)
		self.isImpossibleOutput = nn.Linear(512*256, 2) #TODO: 512 sequence length, da parametrizzare
		self.qaLinear1 = nn.Linear(hiddenSize + 256, hiddenSize + 256)
		self.qaLinear2 = nn.Linear(hiddenSize + 256, 64)
		self.qaOutput = nn.Linear(64, 2)

	def forward(self, inputIDs, sequenceIDs, attentionMask, startPositions=None, endPositions=None):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
#		print("self.bert layer output shape: {}".format(bertOutput.size()))

		midIsImpOutput = self.midIsImpossibleLinear(bertOutput)
#		print("self.midIsImpossibleLinear layer: {} - output shape: {}".format(self.midIsImpossibleLinear, midIsImpOutput.size()))

		batchSize = midIsImpOutput.size()[0] # should be batch size
#		print("Batch size:", batchSize)

		midIsImpOutputFlattened = midIsImpOutput.view(batchSize, 1, -1)
#		print("midIsImpOutputFlattened shape:", midIsImpOutputFlattened.size())

		isImpOutput = self.isImpossibleOutput(midIsImpOutputFlattened)
		isImpOutput = isImpOutput.squeeze()
#		print("self.isImpossibleOutput later: {} - output shape: {}".format(self.isImpossibleOutput, isImpOutput.size()))

		

		concat = torch.cat((bertOutput, midIsImpOutput), dim=-1)
		qaOutput1 = self.qaLinear1(concat)
		qaOutput2 = self.qaLinear2(qaOutput1)
		logits = self.qaOutput(qaOutput2)
		
		#logits = self.qaOutput(bertOutput)
		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits, isImpOutput
"""
		if startPositions is not None and endPositions is not None:
			if len(startPositions.size()) > 1:
				startPositions = startPositions.squeeze(-1)
			if len(endPositions.size()) > 1:
				endPositions = endPositions.squeeze(-1)

			ignoredIndex = startLogits.size(1)
			startPositions.clamp_(0, ignoredIndex)
			endPositions.clamp_(0, ignoredIndex)

			lossFunction = CrossEntropyLoss(ignore_index=ignoredIndex)
			startLoss = lossFunction(startLogits, startPositions)
			endLoss = lossFunction(endLogits, endPositions)
			return (startLoss + endLoss) / 2
		else:
"""



		
