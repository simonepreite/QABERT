#!/usr/bin/env python3

import os
import pickle
import torch
import random
import numpy as np
from BERT import QABERTDebug
from Tokenization import BERTTokenizer
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset, DistributedSampler, RandomSampler)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import namedtuple
from tqdm import tqdm, trange
import argparse
from SQuADDataset import InputFeatures, readSQuADDataset, featurizeExamples, writePredictions, RawResult
from utils import saveVocab
from datetime import datetime


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--outputDir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
	parser.add_argument("--trainFile", default=None, type=str, required=True)
	parser.add_argument("--predictFile", default=None, type=str, required=True)
	parser.add_argument("--vocabFile", default=None, type=str, required=True)
	parser.add_argument("--modelWeights", default=None, type=str, required=True)
	parser.add_argument("--doTraining", default=True, type=bool)
	args = parser.parse_args()
	
	seed = 42
	
	outputDir = args.outputDir
	vocabFile = args.vocabFile
	trainFile = args.trainFile
	predictFile = args.predictFile
	modelWeights = args.modelWeights
	#doTraining = args.doTraining
	doTraining = False
	trainBatchSize = 16
	evalBatchSize = 256
	numTrainEpochs = 2 # these both are parameter that have to come from the commmand line

	#if torch.cuda.is_available()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	nGPU = torch.cuda.device_count()
	
#	torch.cuda.set_device(0)
#	device = torch.device("cuda", 0)
#	nGPU = 1
	# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#	torch.distributed.init_process_group(backend='nccl', rank=0, world_size=2)

	random.seed(seed)
	np.random.seed(seed)
	if nGPU > 0:
		torch.cuda.manual_seed_all(seed)
	
	tokenizer = BERTTokenizer(vocabFile)
	trainExamples = readSQuADDataset(trainFile, True, squadV2=True)
	numTrainOptimizationStep = len(trainExamples) // trainBatchSize * numTrainEpochs
	
	model = QABERTDebug.loadPretrained(modelWeights, False, "", 768)
#	model.to(device)
	if nGPU > 1:
		model = torch.nn.DataParallel(model)
	model.to(device)
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
	allIsImpossible = torch.tensor([f.isImpossible for f in trainFeatures], dtype=torch.float)
	
	trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos, allIsImpossible)
	#trainSampler = DistributedSampler(trainData)
	trainSampler = RandomSampler(trainData)
	trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=trainBatchSize)

	print("Starting dev dataset creation...")
	evalExamples = readSQuADDataset(predictFile, False, squadV2=True)
	print(evalExamples[0].isImpossible)
	print(evalExamples[5].isImpossible)
#	evalFeatures = featurizeExamples(evalExamples, tokenizer, 512, 128, 256, False)

	cachedEvalFeaturesFile = outputDir + "/evalFeatures.bin"
	evalFeatures = None
	try:
		with open(cachedEvalFeaturesFile, "rb") as reader:
			evalFeatures = pickle.load(reader)
	except:
		evalFeatures = featurizeExamples(evalExamples, tokenizer, 512, 128, 256, False) #to generalize paramenters
		with open(cachedEvalFeaturesFile, "wb") as writer:
			pickle.dump(evalFeatures, writer)

	allInputIDs = torch.tensor([f.inputIDs for f in evalFeatures], dtype=torch.long)
	allInputMask = torch.tensor([f.inputMask for f in evalFeatures], dtype=torch.long)
	allSegmentIDs = torch.tensor([f.segmentIDs for f in evalFeatures], dtype=torch.long)
	allExampleIndex = torch.arange(allInputIDs.size(0), dtype=torch.long)
#	allIsImpossible = torch.tensor([f.isImpossible for f in evalFeatures], dtype=torch.long)
	evalData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allExampleIndex)

	evalSampler = SequentialSampler(evalData)
	evalDataLoader = DataLoader(evalData, sampler=evalSampler, batch_size=evalBatchSize)

	if doTraining:
		if nGPU > 1:
			deactivatedLayers = [model.module.bert]
		else:
			deactivatedLayers = [model.bert]
		for l in deactivatedLayers:
			for v in l.parameters():
				v.requires_grad = False


		print("Training...")
		model.train()
	
		print("Start training for qaOutputs part")
		for epoch in trange(int(numTrainEpochs), desc="Epoch"):
			for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration")):
				if nGPU >= 1:
					batch = tuple(t.to(device) for t in batch)

				inputIDs, inputMask, segmentIDs, startPositions, endPositions, isImpossibles = batch
				startLogits, endLogits = model(inputIDs, inputMask, segmentIDs)

				if len(startPositions.size()) > 1:
					startPositions = startPositions.squeeze(-1)
				if len(endPositions.size()) > 1:
					endPositions = endPositions.squeeze(-1)

				ignoredIndex = startLogits.size(1)
				startPositions.clamp_(0, ignoredIndex)
				endPositions.clamp_(0, ignoredIndex)

				lossFun = CrossEntropyLoss(ignore_index=ignoredIndex)
				startLoss = lossFun(startLogits, startPositions)
				endLoss = lossFun(endLogits, endPositions)
				loss = (startLoss + endLoss) / 2

				if nGPU > 1:
					loss = loss.mean()

				loss.backward()

				optimizer.step()
				optimizer.zero_grad()
				globalStep += 1


		modelToSave = model.module if hasattr(model, "module") else model

		print("Saving model...")
		outputModelFile = outputDir + "/{}_{}_QABERTDebugTrained.bin".format(datetime.now().strftime("%Y-%m-%d_%H-%M"), trainBatchSize)
		torch.save(modelToSave.state_dict(), outputModelFile)

		print("Loading finetuned model...")
		model = QABERTDebug.loadPretrained(outputModelFile, False, outputModelFile, 768)

	print("Predicting...")

	model.to(device)
	model.eval()

	allResults = []
	evalResFile = outputDir + "/{}_{}_evalResult.txt".format(datetime.now().strftime("%Y-%m-%d_%H-%M"), evalBatchSize)
#	for inputIDs, inputMask, segmentIDs, exampleIndices, isImpossibles in tqdm(evalDataLoader, desc="Evaluating"):
	for step, batch in enumerate(evalDataLoader):
		print("Executing batch {} of {}...".format(step+1, len(evalDataLoader)))
		inputIDs, inputMask, segmentIDs, exampleIndices = batch

		precision = 0.0
		loss = 0.0
		recall = 0.0
		f1 = 0.0
		accuracy = 0.0

		inputIDs = inputIDs.to(device)
		inputMask = inputMask.to(device)
		segmentIDs = segmentIDs.to(device)

		with torch.no_grad():
			batchStartLogits, batchEndLogits = model(inputIDs, inputMask, segmentIDs)

		for i, exampleIndex in enumerate(exampleIndices):
#			print("Step {} - Preparing example {}...".format(step, i))
			startLogits = batchStartLogits[i].detach().cpu().tolist()
			endLogits = batchEndLogits[i].detach().cpu().tolist()
			evalFeature = evalFeatures[exampleIndex.item()]
			uniqueID = int(evalFeature.ID)
			allResults.append(RawResult(ID=uniqueID, startLogits=startLogits, endLogits=endLogits))

	outputPredFile = os.path.join(outputDir, "predictions.json")
	outputNBestFile = os.path.join(outputDir, "nbest_predictions.json")
	outputNullLogOddsFile = os.path.join(outputDir, "null_odds.json")

	print("Writing predictions...")
	writePredictions(evalExamples, evalFeatures, allResults, 20, 30, True, outputPredFile, outputNBestFile, outputNullLogOddsFile, version_2_with_negative=True, null_score_diff_threshold=0.0)
	"""
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
			if nGPU >= 1:
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
	"""
	
	
if __name__ == "__main__":
	main()

