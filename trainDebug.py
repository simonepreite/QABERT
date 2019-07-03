#!/usr/bin/env python3

import os
import pickle
import torch
import random
import numpy as np
from BERT import QABERT4LGELUSkip, QABERT2LReLUSkip, QABERT2LGELU, QABERT2LTanh, QABERT4L400Tanh, QABERT4L1024Tanh, QABERT4LReLU, QABERTVanilla, QABERTFail, QABERT
from Tokenization import BERTTokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, RandomSampler
from collections import namedtuple
from tqdm import tqdm, trange
import argparse
from SQuADDataset import readSQuADDataset, featurizeExamples, writePredictions, RawResult
import json


def main():
	parser = argparse.ArgumentParser()

	# Required arguments
	parser.add_argument("--outputDir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
	parser.add_argument("--vocabFile", default=None, type=str, required=True)
	parser.add_argument("--modelWeights", default=None, type=str, required=True)

	# Other arguments
	parser.add_argument("--debugOutputDir", default=None, type=str)
	parser.add_argument("--trainFile", default=None, type=str)
	parser.add_argument("--predictFile", default=None, type=str)
	parser.add_argument("--useTFCheckpoint", action="store_true")
	parser.add_argument("--doTrain", action="store_true")
	parser.add_argument("--doPredict", action="store_true")
	parser.add_argument("--trainEpochs", default=1.0, type=float)
	parser.add_argument("--trainBatchSize", default=12, type=int)
	parser.add_argument("--predictBatchSize", default=8, type=int)
	parser.add_argument("--paragraphStride", default=128, type=int)
	parser.add_argument("--maxSeqLength", default=384, type=int)
	parser.add_argument("--maxQueryLength", default=64, type=int)
	parser.add_argument("--useVer2", action="store_true")
	parser.add_argument("--learningRate", default=5e-5, type=float)
	parser.add_argument("--nBestSize", default=20, type=int)
	parser.add_argument("--maxAnswerLength", default=30, type=int)
	parser.add_argument("--doLowercase", action="store_true")
	parser.add_argument("--trainDevFile", default=None, type=str)
	parser.add_argument("--bertTrainable", action="store_true")
	parser.add_argument("--linearShapes", nargs="+", type=int, default=None)
	parser.add_argument("--activationFun", default=None, type=str)
	args = parser.parse_args()
	
	if args.linearShapes:
		args.linearShapes = tuple(args.linearShapes)
	
	if (not args.doTrain) and (not args.doPredict):
		raise Exception("At least one between --doTrain and --doPredict must be True.")
	
	seed = 42
	hiddenSize = 768

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	nGPU = torch.cuda.device_count()

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if nGPU > 0:
		torch.cuda.manual_seed_all(seed)
	
	tokenizer = BERTTokenizer(args.vocabFile, args.doLowercase)

	convertedWeights = ""
	if args.useTFCheckpoint:
		convertedWeights = args.outputDir + "/ptWeights_{}_{}_{}_{}_{}.bin".format("uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength)
		
	model =  QABERT4LGELUSkip.loadPretrained(args.modelWeights, args.useTFCheckpoint, convertedWeights, hiddenSize)
	#model = QABERT.loadPretrained(args.modelWeights, args.useTFCheckpoint, convertedWeights, hiddenSize, shapes=args.linearShapes, activationFun=args.activationFun)
	
	if args.debugOutputDir:
		with open(args.debugOutputDir + "/modelSummary.txt", "w") as file:
			print(model, file=file)
	
	model.to(device)

	if nGPU > 1:
		model = torch.nn.DataParallel(model)

	#TODO: optimizer has to be implemented
	optimizer = Adam(model.parameters(), lr=args.learningRate)

	print("Starting featurization...")
	globalStep = 0

	if args.doTrain:
		trainExamples = readSQuADDataset(args.trainFile, True, squadV2=args.useVer2)

		if args.debugOutputDir:
			with open(args.debugOutputDir + "/trainExamplesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in trainExamples], indent=2), file=file)

		# TODO: to be implemented
		numTrainOptimizationStep = len(trainExamples) // args.trainBatchSize * args.trainEpochs

		cachedTrainFeaturesFile = args.outputDir + "/trainFeatures_{}_{}_{}_{}_{}.bin".format("uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength)

		trainFeatures = None
		try:
			with open(cachedTrainFeaturesFile, "rb") as reader:
				trainFeatures = pickle.load(reader)
		except:
			print("Building train features...")
			trainFeatures = featurizeExamples(trainExamples, tokenizer, args.maxSeqLength, args.paragraphStride, args.maxQueryLength, True)
			with open(cachedTrainFeaturesFile, "wb") as writer:
				pickle.dump(trainFeatures, writer)

		if args.debugOutputDir:
			with open(args.debugOutputDir + "/trainFeaturesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in trainFeatures]), file=file)


		print("Starting train dataset creation...")
		allInputIDs = torch.tensor([f.inputIDs for f in trainFeatures], dtype=torch.long)
		allInputMask = torch.tensor([f.inputMask for f in trainFeatures], dtype=torch.long)
		allSegmentIDs = torch.tensor([f.segmentIDs for f in trainFeatures], dtype=torch.long)
		allStartPos = torch.tensor([f.startPos for f in trainFeatures], dtype=torch.long)
		allEndPos = torch.tensor([f.endPos for f in trainFeatures], dtype=torch.long)
		#allIsImpossible = torch.tensor([f.isImpossible for f in trainFeatures], dtype=torch.float)

		trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos)
#		trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos, allIsImpossible)
		
		trainSampler = RandomSampler(trainData)
		trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=args.trainBatchSize)
		if args.trainDevFile:
			print("Starting train-dev dataset creation...")
			trainDevExamples = readSQuADDataset(args.trainDevFile, False, squadV2=args.useVer2)

			if args.debugOutputDir:
				with open(args.debugOutputDir + "/trainDevExamplesDebug.json", "w") as file:
					print(json.dumps([t._asdict() for t in trainDevExamples], indent=2), file=file)

			cachedTrainDevFeaturesFile = args.outputDir + "/trainDevFeatures_{}_{}_{}_{}_{}.bin".format("uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength)

			trainDevFeatures = None
			try:
				with open(cachedTrainDevFeaturesFile, "rb") as reader:
					trainDevFeatures = pickle.load(reader)
			except:
				print("Building train-dev features...")
				trainDevFeatures = featurizeExamples(trainDevExamples, tokenizer, args.maxSeqLength, args.paragraphStride, args.maxQueryLength, False)
				with open(cachedTrainDevFeaturesFile, "wb") as writer:
					pickle.dump(trainDevFeatures, writer)

			if args.debugOutputDir:
				with open(args.debugOutputDir + "/trainDevFeaturesDebug.json", "w") as file:
					print(json.dumps([t._asdict() for t in trainDevFeatures], indent=2), file=file)

			allInputIDsTD = torch.tensor([f.inputIDs for f in trainDevFeatures], dtype=torch.long)
			allInputMaskTD = torch.tensor([f.inputMask for f in trainDevFeatures], dtype=torch.long)
			allSegmentIDsTD = torch.tensor([f.segmentIDs for f in trainDevFeatures], dtype=torch.long)
			allExampleIndexTD = torch.arange(allInputIDsTD.size(0), dtype=torch.long)
			#		allIsImpossible = torch.tensor([f.isImpossible for f in evalFeatures], dtype=torch.long)
			trainDevData = TensorDataset(allInputIDsTD, allInputMaskTD, allSegmentIDsTD, allExampleIndexTD)

			trainDevSampler = SequentialSampler(trainDevData)
			trainDevDataLoader = DataLoader(trainDevData, sampler=trainDevSampler, batch_size=args.predictBatchSize)

	if args.doPredict:
		print("Starting dev dataset creation...")
		evalExamples = readSQuADDataset(args.predictFile, False, squadV2=args.useVer2)

		if args.debugOutputDir:
			with open(args.debugOutputDir + "/evalExamplesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in evalExamples], indent=2), file=file)

		cachedEvalFeaturesFile = args.outputDir + "/evalFeatures_{}_{}_{}_{}_{}.bin".format("uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength)

		evalFeatures = None
		try:
			with open(cachedEvalFeaturesFile, "rb") as reader:
				evalFeatures = pickle.load(reader)
		except:
			print("Building eval features...")
			evalFeatures = featurizeExamples(evalExamples, tokenizer, args.maxSeqLength, args.paragraphStride, args.maxQueryLength, False)
			with open(cachedEvalFeaturesFile, "wb") as writer:
				pickle.dump(evalFeatures, writer)

		if args.debugOutputDir:
			with open(args.debugOutputDir + "/evalFeaturesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in evalFeatures], indent=2), file=file)

		allInputIDs = torch.tensor([f.inputIDs for f in evalFeatures], dtype=torch.long)
		allInputMask = torch.tensor([f.inputMask for f in evalFeatures], dtype=torch.long)
		allSegmentIDs = torch.tensor([f.segmentIDs for f in evalFeatures], dtype=torch.long)
		allExampleIndex = torch.arange(allInputIDs.size(0), dtype=torch.long)
#		allIsImpossible = torch.tensor([f.isImpossible for f in evalFeatures], dtype=torch.long)
		evalData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allExampleIndex)

		evalSampler = SequentialSampler(evalData)
		evalDataLoader = DataLoader(evalData, sampler=evalSampler, batch_size=args.predictBatchSize)

	if args.doTrain:
		if not args.bertTrainable:
			noFineTuningLayers = [model.module.bert] if nGPU > 1 else [model.bert]

			for l in noFineTuningLayers:
				for v in l.parameters():
					v.requires_grad = False


		print("Training...")
		model.train()

		for epoch in trange(int(args.trainEpochs), desc="Epoch"):
			epochLosses = []
			for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration")):
				if nGPU >= 1:
					batch = tuple(t.to(device) for t in batch)

#				inputIDs, inputMask, segmentIDs, startPositions, endPositions, isImpossibles = batch
				inputIDs, inputMask, segmentIDs, startPositions, endPositions = batch
				startLogits, endLogits = model(inputIDs, segmentIDs, inputMask)

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

				epochLosses.append(loss.item())

				arg = "a"
				if epoch == 0 and step == 0:
					arg = "w"
				with open(args.outputDir + "/losstrace.txt", arg) as file:
					print(loss.item(), file=file)

				optimizer.step()
				optimizer.zero_grad()
				globalStep += 1

			arg = "a"
			if epoch == 0:
				arg = "w"
			with open(args.outputDir + "/lossEpochsTrace.txt", arg) as file:
				print(torch.mean(torch.tensor(epochLosses)).item(), file=file)
		
		modelToSave = model.module if hasattr(model, "module") else model

		print("Saving model...")
		outputModelFile = args.outputDir + "/QABERTTrained_{}_{}_{}_{}_{}_{}_{}.bin".format("uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength, args.trainBatchSize, args.trainEpochs)
		torch.save(modelToSave.state_dict(), outputModelFile)

		print("Loading finetuned model...")
		model = QABERT4LGELUSkip.loadPretrained(outputModelFile, False, "", hiddenSize)
		#model = QABERT.loadPretrained(outputModelFile, False, "", hiddenSize, shapes=args.linearShapes, activationFun=args.activationFun)


	if args.doPredict:
		print("Predicting...")
		model.to(device)
		model.eval()

		allResults = []
		for step, batch in enumerate(tqdm(evalDataLoader, desc="Batches")):

			inputIDs, inputMask, segmentIDs, exampleIndices = batch

			inputIDs = inputIDs.to(device)
			inputMask = inputMask.to(device)
			segmentIDs = segmentIDs.to(device)

			with torch.no_grad():
				batchStartLogits, batchEndLogits = model(inputIDs, segmentIDs, inputMask)

			for i, exampleIndex in enumerate(exampleIndices):
				startLogits = batchStartLogits[i].detach().cpu().tolist()
				endLogits = batchEndLogits[i].detach().cpu().tolist()
				evalFeature = evalFeatures[exampleIndex.item()]
				uniqueID = int(evalFeature.ID)
				allResults.append(RawResult(ID=uniqueID, startLogits=startLogits, endLogits=endLogits))

		outputPredFile = os.path.join(args.outputDir, "predictions.json")
		outputNBestFile = os.path.join(args.outputDir, "nbest_predictions.json")
		outputNullLogOddsFile = os.path.join(args.outputDir, "null_odds.json")

		print("Writing predictions...")
		writePredictions(evalExamples, evalFeatures, allResults, args.nBestSize, args.maxAnswerLength, args.doLowercase, outputPredFile, outputNBestFile, outputNullLogOddsFile, args.useVer2, 0.0)

	
	
if __name__ == "__main__":
	main()

