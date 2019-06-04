import torch
from torch import nn
from NormLayer import NormLayer
from utils import loadModuleParameters
from Embeddings import BERTEmbeddings
from Encoder import Encoder


class BERTInitializer(nn.Module):
	def __init__(self, *inputs, **kwargs):
		super(BERTInitializer, self).__init__()
		print("BERTInitializer init")
		print(inputs)
		print(kwargs)

	def weightsInitialization(self, module):
		print("weightsInitialization")
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# implement truncated normal for weights init
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, NormLayer):
			module.bias.data.zero_()
			module.weights.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	@classmethod
	def loadPretrained(className, checkpointPath, *inputs, **kwargs):
		print(inputs)
		print("loadPretrained")
		stateDict = kwargs.get("state_dict", None)
		kwargs.pop("state_dict", None)

		model = className(*inputs, **kwargs)

		if stateDict is None:
			stateDict = torch.load(checkpointPath, map_location="cpu")

		oldKeys = []
		newKeys = []
		for key in stateDict.keys():
			newKey = None
			if "gamma" in key:
				newKey = key.replace("gamma", "weight")
			if "beta" in key:
				newKey = key.replace("beta", "bias")
			if newKey:
				oldKeys.append(key)
				newKeys.append(newKey)
		for oK, nK in zip(oldKeys, newKeys):
			stateDict[nK] = stateDict.pop(oK)

		missingKeys = []
		unexpKeys = []
		errors = []

		metadata = getattr(stateDict, "_metadata", None)
		stateDict = stateDict.copy()
		if metadata is not None:
			stateDict._metadata = metadata

		prefix = ""
		# if not hasattr(model, "bert") and any(s.startswith("bert.") for s in stateDict.keys()):
		# 	prefix = "bert."
		#print(stateDict.keys())
		#print(stateDict["encoder.0.0.feedForward.w1.weight"])
		stateDict["encoder.0.0.feedForward.w1.weight"] = stateDict["bert.encoder.layer.0.attention.output.dense.weight"]
		print(stateDict["encoder.0.0.feedForward.w1.weight"])
		#print(model.state_dict().keys())
		loadModuleParameters(model, prefix, metadata, state_dict=stateDict, missing_keys=missingKeys, unexp_keys=unexpKeys, errors=errors)

		print(stateDict["encoder.0.0.feedForward.w1.weight"])
		print(missingKeys)
		print()
		print(unexpKeys)
		print()
		print(errors)

		return model


class BERTModel(BERTInitializer):
	def __init__(self, hiddenSize, numLayers=12, numAttentionHeads=12, vocabSize=30522, dropout=0.1):
		print("BERT INIT")
		super(BERTModel, self).__init__(hiddenSize, numLayers, numAttentionHeads, vocabSize, dropout)
		self.hiddenSize = hiddenSize
		self.numLayers = numLayers
		self.numAttentionHeads = numAttentionHeads
		self.maxPosEmbedding = 512
		self.vocabSize = vocabSize

		#self.apply(self.weightsInitialization)
		
		self.embeddings = BERTEmbeddings(self.hiddenSize, self.vocabSize, self.maxPosEmbedding)
		self.encoder = nn.Sequential(nn.ModuleList(Encoder(hiddenSize, numAttentionHeads) for _ in range(numLayers)))
		
		
	def forward(tensor, sequenceIDs):
		print("forward")
		x = torch.FloatTensor([[[ 0.9059, -0.7039, -0.3376,  0.1968],[-1.0413,  0.8128,  0.0697, -0.6166],[-0.3793, -0.9851, -2.3841, -0.7003],[ 0.6076, -1.4874, -0.1079,  0.4266]]])

		inputIDs = torch.randn(1, 4)
		attentionMask = torch.ones_like(inputIDs)
		extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2)
		#extendedAttentionMask = extendedAttentionMask.to(dtype=next(self.parameters()).dtype)
		extendedAttentionMask = (1.0 - extendedAttentionMask) * -10000.0
		
		x = self.embeddings(x, sequenceIDs)
		x = self.encoder(x, extendedAttentionMask)




		