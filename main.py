#!/usr/bin/env python3
import pickle
import torch
import random
import numpy as np
from BERT import QABERT
from Tokenization import BERTTokenizer
from SQuADDataset import featurizeExamples, readSQuADDataset
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset, DistributedSampler, RandomSampler)
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import namedtuple
from tqdm import tqdm, trange
import argparse
from SQuADDataset import InputFeatures


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--outputDir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
	parser.add_argument("--trainFile", default=None, type=str, required=True)
	parser.add_argument("--predictFile", default=None, type=str, required=True)
	parser.add_argument("--vocabFile", default=None, type=str, required=True)
	parser.add_argument("--modelWeights", default=None, type=str, required=True)
	args = parser.parse_args()
	
	seed = 42
	
	outputDir = args.outputDir
	vocabFile = args.vocabFile
	trainFile = args.trainFile
	predictFile = args.predictFile
	modelWeights = args.modelWeights
	trainBatchSize = 12
	numTrainEpochs = 1 # these both are parameter that have to come from the commmand line
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	nGPU = torch.cuda.device_count()
	#torch.distributed.init_process_group(backend='nccl')

	random.seed(seed)
	np.random.seed(seed)
	if nGPU > 0:
		torch.cuda.manual_seed_all(seed)
	
	tokenizer = BERTTokenizer(vocabFile)
	trainExamples = readSQuADDataset(trainFile, True, squadV2=True)
	numTrainOptimizationStep = len(trainExamples) // trainBatchSize * numTrainEpochs
	model = QABERT.loadPretrained(modelWeights, False, "", 768)
	model.to(device)
	if nGPU > 1:
		model = torch.nn.DataParallel(model)
	
	# no_decay = ['bias', 'NormLayer.bias', 'NormLayer.weight']
	# optimizer_grouped_parameters = [
	# 	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	# 	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	# 	]
	
	#TODO: optimizer has to be implemented
	optimizer = Adam(model.parameters(), lr=3e-5)
	
	print("Starting featurization...")
	globalStep = 0
	cachedTrainFeaturesFile = outputDir + "/trainFeatures.bin"
	trainFeatures = None
	try:
		with open(cachedTrainFeaturesFile, "rb") as reader:
			trainFeatures = pickle.load(reader)
	except:
		trainFeatures = featurizeExamples(trainExamples, tokenizer, 512, 128, 256, True) #to generalize paramenters
		with open(cachedTrainFeaturesFile, "wb") as writer:
			pickle.dump(trainFeatures, writer)
	
	print("Starting tensor dataset creation...")
	allInputIDs = torch.tensor([f.inputIDs for f in trainFeatures], dtype=torch.long)
	allInputMask = torch.tensor([f.inputMask for f in trainFeatures], dtype=torch.long)
	allSegmentIDs = torch.tensor([f.segmentIDs for f in trainFeatures], dtype=torch.long)
	allStartPos = torch.tensor([f.startPos for f in trainFeatures], dtype=torch.long)
	allEndPos = torch.tensor([f.endPos for f in trainFeatures], dtype=torch.long)
	allIsImpossible = torch.tensor([f.isImpossible for f in trainFeatures], dtype=torch.long)
	
	trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos, allIsImpossible)
	#trainSampler = DistributedSampler(trainData)
	trainSampler = RandomSampler(trainData)
	trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=trainBatchSize)

	print("Starting dev dataset creation...")
	evalExamples = readSQuADDataset(predictFile, False, squadV2=True)
	evalFeatures = featurizeExamples(evalExamples, tokenizer, 512, 128, 256, False)

	allInputIDs = torch.tensor([f.inputIDs for f in evalFeatures], dtype=torch.long)
	allInputMask = torch.tensor([f.inputMask for f in evalFeatures], dtype=torch.long)
	allSegmentIDs = torch.tensor([f.segmentIDs for f in evalFeatures], dtype=torch.long)
	allExampleIndex = torch.arange(allInputIDs.size(0), dtype=torch.long)
	allIsImpossible = torch.tensor([f.isImpossible for f in evalFeatures], dtype=torch.long)
	evalData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allExampleIndex, allIsImpossible)

	evalSampler = SequentialSampler(evalData)
	evalDataLoader = DataLoader(evalData, sampler=evalSampler, batch_size=trainBatchSize)

	evalBatchInputIDs, evalBatchInputMask, evalBatchSegmentIDs, evalBatchExampleIndices, evalIsImpossibles = next(iter(evalDataLoader))
	evalBatchInputIDs = evalBatchInputIDs.to(device)
	evalBatchInputMask = evalBatchInputMask.to(device)
	evalBatchSegmentIDs = evalBatchSegmentIDs.to(device)
	evalIsImpossibles = evalIsImpossibles.to(device)




	print("Training...")
	model.train()

	deactivatedLayers = [model.bert, model.qaLinear1, model.qaLinear2, model.qaOutput]
	for l in deactivatedLayers:
		for v in l.parameters():
			v.requires_grad = False
	
	print("Start training for isImpossible part")
	# Training for the isImpossible part of the network
	for epoch in range(int(numTrainEpochs)):
		for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration for IsImpossible")):
			if nGPU == 1:
				batch = tuple(t.to(device) for t in batch)
			print(batch)
			print("\n")
			inputIDs, inputMask, segmentIDs, startPositions, endPositions, isImpossibles = batch
			print("inputIDs: {}, segmentIDs: {}, isImpossibles: {}".format(inputIDs.size(), segmentIDs.size(), isImpossibles.size()))
			_, _, isImpossibleComputed = model(inputIDs, inputMask, segmentIDs)
			print("isImpossibleComputed: {}".format(isImpossibleComputed.size()))

			classWeights = torch.tensor([1, 2])
			weightedLossFun = BCELoss(weight=classWeights)
			loss = weightedLossFun(isImpossibleComputed, isImpossibles)

			if nGPU > 1:
				loss = loss.mean()

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()
			globalStep += 1

			precision = precision_score(isImpossibles, isImpossibleComputed)
			recall = recall_score(isImpossibles, isImpossibleComputed)
			f1 = f1_score(isImpossibles, isImpossibleComputed)

			print("Step: {} - Loss: {}, Precision: {}, Recall: {}, F1: {}".format(loss, step, precision, recall, f1), end="\r")

		with torch.no_grad():
			batchStartLogits, batchEndLogits, batchIsImpossible = model(evalBatchInputIDs, evalBatchInputMask, evalBatchSegmentIDs)

		precision = precision_score(evalIsImpossibles, batchIsImpossible)
		recall = recall_score(evalIsImpossibles, batchIsImpossible)
		f1 = f1_score(evalIsImpossibles, batchIsImpossible)

		print("Epoch: {} - Precision: {}, Recall: {}, F1: {}".format(epoch, precision, recall, f1))

	for l in deactivatedLayers:
		for v in l.parameters():
			v.requires_grad = True

	deactivatedLayers = [model.bert, model.midIsImpossibleLinear, model.isImpossibleOutput]
	for l in deactivatedLayers:
		for v in l.parameters():
			v.requires_grad = False

	# Training for the QA part of the network
	for _ in trange(int(numTrainEpochs), desc="QA Epoch"):
		for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration for QA")):
			if nGPU == 1:
				batch = tuple(t.to(device) for t in batch)

			inputIDs, inputMask, segmentIDs, startPositions, endPositions, isImpossibles = batch
			startLogits, endLogits, _ = model(inputIDs, inputMask, segmentIDs)

			if len(startPositions.size()) > 1:
				startPositions = startPositions.squeeze(-1)
			if len(endPositions.size()) > 1:
				endPositions = endPositions.squeeze(-1)

			ignoredIndexes = startLogits.size(1)
			startPositions.clamp_(0, ignoredIndexes)
			endPositions.clamp_(0, ignoredIndexes)

			lossFun = CrossEntropyLoss(ignore_index=ignoredIndexes)
			startLoss = lossFun(startLogits, startPositions)
			endLoss = lossFun(endLogits, endPositions)
			loss = (startLoss + endLoss) / 2
	
			if nGPU > 1:
				loss = loss.mean()

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()
			globalStep += 1
	
	model.to(device)

	print("Predicting...")
	model.eval()
	allResults = []

	for inputIDs, inputMask, segmentIDs, exampleIndices in tqdm(evalDataLoader, desc="Evaluating"):
		if len(allResults) % 1000 == 0:
			logger.info("Processing example: %d" % (len(allResults)))

		inputIDs = inputIDs.to(device)
		inputMask = inputMask.to(device)
		segmentIDs = segmentIDs.to(device)

		with torch.no_grad():
			batchStartLogits, batchEndLogits, batchIsImpossible = model(inputIDs, inputMask, segmentIDs)

		for i, exampleIndex in enumerate(exampleIndices):
			startLogits = batchStartLogits[i].detach().cpu().tolist()
			endLogits = batchEndLogits[i].detach().cpu().tolist()
			evalFeature = evalFeatures[exampleIndex.item()]
			uniqueID = int(evalFeature.ID)
			allResults.append({"unique_id": uniqueID, "start_logits": startLogits, "end_logits": endLogits})

		outputPredFile = os.path.join(outputDir, "predictions.json")
		outputNBestFile = os.path.join(outputDir, "nbest_predictions.json")
		outputNullLogOddsFile = os.path.join(outputDir, "null_odds.json")

		writePredictions(evalExamples, evalFeatures, allResults, 20, 30, True, outputPredFile, outputNBestFile, outputNullLogOddsFile, usingV2=True, nullDiffThreshold=0.0)

	
	
if __name__ == "__main__":
	main()

