import torch
import copy
from torch import nn
from NormLayer import NormLayer
from utils import loadModuleParameters
from Embeddings import BERTEmbeddings
from Encoder import Encoder
from convertWeights import convertTensorFlowWeights
from GELU import GELU


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
		
		self.embeddings = BERTEmbeddings(self.hiddenSize, self.vocabSize, self.maxPosEmbedding)

		encoder = Encoder(hiddenSize, numAttentionHeads)
		self.encoder = nn.ModuleList([copy.deepcopy(encoder) for _ in range(numLayers)])
		
		self.apply(self.weightsInitialization)
		
		
	def forward(self, inputIDs, sequenceIDs, attentionMask=None):
		if attentionMask is None:
			attentionMask = torch.ones_like(inputIDs)
		extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2)
		extendedAttentionMask = extendedAttentionMask.to(dtype=next(self.parameters()).dtype)
		extendedAttentionMask = (1.0 - extendedAttentionMask) * -10000.0
		
		embeddingOutput = self.embeddings(inputIDs, sequenceIDs)
		encodedLayers = []

		for i, layer in enumerate(self.encoder):
			if i == 0:
				hiddenStates = layer(embeddingOutput, extendedAttentionMask)
			else:
				hiddenStates = layer(hiddenStates, extendedAttentionMask)
			encodedLayers.append(hiddenStates)

		return encodedLayers[-1]

class QABERT2LGELU(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT2LGELU, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.middleOutput1 = nn.Linear(768, 256)
		self.qaOutputs = nn.Linear(256, 2)
		self.activationFun = GELU()
		self.dropout = nn.Dropout(dropout)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		middleOutput1 = self.dropout(self.activationFun(self.middleOutput1(bertOutput)))
		logits = self.qaOutputs(middleOutput1)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits

class QABERT2LTanh(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT2LTanh, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.middleOutput = nn.Linear(768, 384)
		self.qaOutputs = nn.Linear(384, 2)
		self.activationFun = nn.Tanh()
		self.dropout = nn.Dropout(dropout)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		middleOutput1 = self.dropout(self.activationFun(self.middleOutput(bertOutput)))
		logits = self.qaOutputs(middleOutput1)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits
		

class QABERT4LTanh(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT4L1024Tanh, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.middleOutput1 = nn.Linear(768, 1024)
		self.middleOutput2 = nn.Linear(1024, 200)
		self.middleOutput3 = nn.Linear(200, 64)
		self.qaOutputs = nn.Linear(64, 2)
		self.activationFun = nn.Tanh()
		self.dropout = nn.Dropout(dropout)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		middleOutput1 = self.dropout(self.activationFun(self.middleOutput1(bertOutput)))
		middleOutput2 = self.dropout(self.activationFun(self.middleOutput2(middleOutput1)))
		middleOutput3 = self.dropout(self.activationFun(self.middleOutput3(middleOutput2)))
		logits = self.qaOutputs(middleOutput3)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits

class QABERT4LReLU(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT4LReLU, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.middleOutput1 = nn.Linear(768, 1024)
		self.middleOutput2 = nn.Linear(1024, 200)
		self.middleOutput3 = nn.Linear(200, 64)
		self.qaOutputs = nn.Linear(64, 2)
		self.activationFun = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		middleOutput1 = self.dropout(self.activationFun(self.middleOutput1(bertOutput)))
		middleOutput2 = self.dropout(self.activationFun(self.middleOutput2(middleOutput1)))
		middleOutput3 = self.dropout(self.activationFun(self.middleOutput3(middleOutput2)))
		logits = self.qaOutputs(middleOutput3)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits

class QABERTVanilla(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERTVanilla, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.qaOutputs = nn.Linear(hiddenSize, 2)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		logits = self.qaOutputs(bertOutput)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits
		
		
class QABERT2LReLUSkip(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT2LReLUSkip, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.middleOutput1 = nn.Linear(768, 768)
		self.qaOutputs = nn.Linear(768, 2)
		self.activationFun = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		middleOutput1 = self.dropout(self.activationFun(self.middleOutput1(bertOutput)))
		logits = self.qaOutputs(middleOutput1 + bertOutput)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits


class QABERT4LGELUSkip(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		super(QABERT4LGELUSkip, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)

		self.bert = BERTModel(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.middleOutput1 = nn.Linear(768, 1024)
		self.middleOutput2 = nn.Linear(1024, 768)
		self.middleOutput3 = nn.Linear(768, 384)
		self.qaOutputs = nn.Linear(384, 2)
		self.activationFun = GELU()
		self.dropout = nn.Dropout(dropout)
		self.apply(self.weightsInitialization)

	def forward(self, inputIDs, sequenceIDs, attentionMask):
		bertOutput = self.bert(inputIDs, sequenceIDs, attentionMask)
		middleOutput1 = self.dropout(self.activationFun(self.middleOutput1(bertOutput)))
		middleOutput2 = self.dropout(self.activationFun(self.middleOutput2(middleOutput1)))
		middleOutput3 = self.dropout(self.activationFun(self.middleOutput3(middleOutput2 + bertOutput)))
		logits = self.qaOutputs(middleOutput3)

		startLogits, endLogits = logits.split(1, dim=-1)
		startLogits = startLogits.squeeze(-1)
		endLogits = endLogits.squeeze(-1)

		return startLogits, endLogits