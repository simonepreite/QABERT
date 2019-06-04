#!/usr/bin/env python3

import os
import torch
from BERT import BERTModel

try:
	import re
	import numpy as np
	import tensorflow as tf
except ImportError:
	print("TensorFlow is required to convert checkpoints for PyTorch model.")
	raise

def loadTensorFlowWeights(model, checkpointPath):
	checkpointPath = os.path.abspath(checkpointPath)

	print("Loading TensorFlow checkpoint file...")
	tfVars = tf.train.list_variables(checkpointPath)
	data = dict()

	oldEndings = ["/beta", "/gamma", "/position_embeddings", "/token_type_embeddings", "/word_embeddings", "attention/output/LayerNorm", "attention/output", "attention/self", "intermediate/dense", "output/dense", "output/LayerNorm", "dense", "embeddings/LayerNorm", "key", "query", "value"]
	newEndings = ["/bias", "/weight", "/posEmbeddings/weight", "/seqEmbeddings/weight", "/wordEmbeddings/weight", "attNorm", "multiHeadAtt", "multiHeadAtt", "feedForward/w1", "feedForward/w2", "outputNorm", "outputLinear", "embeddings/normLayer", "keysLinear", "queriesLinear", "valuesLinear"]

	for name, shape in tfVars:
		w = tf.train.load_variable(checkpointPath, name)
		name = name.replace("bert/", "")
		for old, new in zip(oldEndings, newEndings):
			if old in name:
				name = name.replace(old, new)
		data[name] = w

	filteredDict = dict(filter(lambda x: x[0].startswith("embeddings") or x[0].startswith("encoder"), data.items()))

	for key, value in filteredDict.items():
		modelPtr = model
 
		for elem in key.split("/"):
			if re.fullmatch(r'[A-Za-z]+_\d+', elem):
				modelPtr = getattr(modelPtr, '0')
				split = re.split(r'_(\d+)', elem)
				if (len(split) > 1):
					layerNumber = split[1]
					modelPtr = getattr(modelPtr, layerNumber)
			else:
				if elem == "kernel":
					elem = "weight"
					value = np.transpose(value)
				try:
					modelPtr = getattr(modelPtr, elem)
				except AttributeError:
					print("Skipping {}".format(key))
					continue

		try:
			assert modelPtr.shape == value.shape
		except AssertionError as e:
			e.args += (modelPtr.shape, value.shape)
			raise

		print("Converting to PyTorch weights: {}".format(key))
		modelPtr.data = torch.from_numpy(value)

	# Test Code
	# print("After loading...")
	# pprint(list(filteredDict.keys()))
	# stateDict = model.state_dict()
	# for key, value in filteredDict.items():
	# 	if "kernel" in key:
	# 		value = np.transpose(value)
	# 	myKey = key.replace("layer_", "0.").replace("/", ".").replace("kernel", "weight")
	# 	loaded = stateDict[myKey].numpy()
	# 	print("Comparing checkpoint value and loaded for {}; equal? {}".format(myKey, np.array_equal(value, loaded)))

	# loadedArray = model.encoder[0][0].multiHeadAtt.outputLinear.weight.detach().numpy()
	# checkpointArray = data["encoder/layer_0/multiHeadAtt/outputLinear/kernel"]
	# print(loadedArray.shape, checkpointArray.shape)
	# print(np.array_equal(loadedArray, np.transpose(checkpointArray)))


# Test Code
path = "../BERT Checkpoints Tensorflow/Bert Base Uncased/bert_model.ckpt"
model = BERTModel(768)
loadTensorFlowWeights(model, path)

