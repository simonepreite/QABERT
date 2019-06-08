#!/usr/bin/env python3

import os
import torch

try:
	import re
	import numpy as np
	import tensorflow as tf
except ImportError:
	print("TensorFlow is required to convert checkpoints for PyTorch model.")
	raise

def convertTensorFlowWeights(model, checkpointPath, outputPath=None):
	checkpointPath = os.path.abspath(checkpointPath)

	print("Loading TensorFlow checkpoint file...")
	tfVars = tf.train.list_variables(checkpointPath)
	data = dict()

	oldEndings = ["/beta", "/gamma", "/position_embeddings", "/token_type_embeddings", "/word_embeddings", "attention/output/LayerNorm", "attention/output", "attention/self", "intermediate/dense", "output/dense", "output/LayerNorm", "dense", "embeddings/LayerNorm", "key", "query", "value"]
	newEndings = ["/bias", "/weight", "/posEmbeddings/weight", "/seqEmbeddings/weight", "/wordEmbeddings/weight", "attNorm", "multiHeadAtt", "multiHeadAtt", "feedForward/w1", "feedForward/w2", "outputNorm", "outputLinear", "embeddings/normLayer", "keysLinear", "queriesLinear", "valuesLinear"]

	for name, shape in tfVars:
		w = tf.train.load_variable(checkpointPath, name)
		# TODO: Fix, this works only for QABERT
		# if isinstance(model, BERTModel): 
		# 	name = name.replace("bert/", "")
		for old, new in zip(oldEndings, newEndings):
			if old in name:
				name = name.replace(old, new)
		data[name] = w

	print(np.transpose(data["bert/encoder/layer_0/multiHeadAtt/outputLinear/kernel"]))

	filteredDict = dict(filter(lambda x: x[0].startswith("bert/embeddings") or x[0].startswith("bert/encoder"), data.items()))

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

	if outputPath:
		print("Saving PyTorch model weights to {}".format(outputPath))
		torch.save(model.state_dict(), outputPath)



