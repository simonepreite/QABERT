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
from optimizer import BertAdam
from knockknock import telegram_sender

token=""
chat=None

@telegram_sender(token=token, chat_id=chat)
def main(outputDir, vocabFile, modelWeights, trainFile, predictFile, useTFCheckpoint, doTrain, doPredict, trainEpochs, trainBatchSize, predictBatchSize, paragraphStride, maxSeqLength, maxQueryLength, useVer2, learningRate, nBestSize, maxAnswerLength, doLowercase, useTrainDev, bertTrainable, useDebug, linearShapes, activationFun, model):

	models = {
		"QABERT4LGELUSkip":QABERT4LGELUSkip,
		"QABERT2LReLUSkip":QABERT2LReLUSkip,
		"QABERT2LGELU":QABERT2LGELU,
		"QABERT2LTanh":QABERT2LTanh,
		"QABERT4L400Tanh":QABERT4L400Tanh,
		"QABERT4L1024Tanh":QABERT4L1024Tanh,
		"QABERT4LReLU":QABERT4LReLU,
		"QABERTVanilla":QABERTVanilla
	}

	args = {
		"outputDir": outputDir,
		"vocabFile": vocabFile,
		"modelWeights": modelWeights,
		"trainFile": trainFile,
		"predictFile": predictFile,
		"useTFCheckpoint": useTFCheckpoint,
		"doTrain": doTrain,
		"doPredict": doPredict,
		"trainEpochs": trainEpochs,
		"trainBatchSize": trainBatchSize,
		"predictBatchSize": predictBatchSize,
		"paragraphStride": paragraphStride,
		"maxSeqLength": maxSeqLength,
		"maxQueryLength": maxQueryLength,
		"useVer2": useVer2,
		"learningRate": learningRate,
		"nBestSize": nBestSize,
		"maxAnswerLength": maxAnswerLength,
		"doLowercase": doLowercase,
		"useTrainDev": useTrainDev,
		"bertTrainable": bertTrainable,
		"useDebug": useDebug,
		"model": model
	}
	
	if args.modelName not in models:
		raise Exception("Wrong model name.")

	if (not doTrain) and (not doPredict):
		raise Exception("At least one between --doTrain and --doPredict must be True.")

	if useDebug:
		if not os.path.isdir(outputDir + "/debug"):
			os.mkdir(outputDir + "/debug", 0o755)

	seed = 42
	hiddenSize = 768

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	nGPU = torch.cuda.device_count()

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if nGPU > 0:
		torch.cuda.manual_seed_all(seed)

	tokenizer = BERTTokenizer(vocabFile, doLowercase)

	convertedWeights = ""
	if useTFCheckpoint:
		convertedWeights = outputDir + "/ptWeights_{}_{}_{}_{}_{}.bin".format("uncased" if doLowercase else "cased", hiddenSize, maxSeqLength, paragraphStride, maxQueryLength)

	model =  models[model].loadPretrained(modelWeights, useTFCheckpoint, convertedWeights, hiddenSize)
	#model = QABERT.loadPretrained(modelWeights, useTFCheckpoint, convertedWeights, hiddenSize, shapes=linearShapes, activationFun=activationFun)

	with open(outputDir + "/modelSummary.txt", "w") as file:
		print(model, file=file)

	model.to(device)

	if nGPU > 1:
		model = torch.nn.DataParallel(model)

	print("Starting featurization...")

	if doTrain:
		with open(outputDir + "/trainParams.txt", "w") as file:
			print(args, file=file)

		trainExamples = readSQuADDataset(trainFile, True, squadV2=useVer2)

		if useDebug:
			with open(outputDir + "/debug/trainExamplesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in trainExamples], indent=2), file=file)

		numTrainOptimizationStep = len(trainExamples) * trainEpochs

		paramOptimizer = list(model.named_parameters())
		noDecay = ["bias", "NormLayer.bias", "NormLayer.weight"]
		parameters = [
			{"params" : [p for n, p in paramOptimizer if not any(nd in n for nd in noDecay)], "weight_decay": 0.01},
			{"params" : [p for n, p in paramOptimizer if any(nd in n for nd in noDecay)], "weight_decay": 0.0}		
		]
		optimizer = BertAdam(parameters, lr=learningRate, warmup=0.1, t_total=numTrainOptimizationStep)


		cachedTrainFeaturesFile = outputDir + "/trainFeatures_{}_{}_{}_{}_{}.bin".format("uncased" if doLowercase else "cased", hiddenSize, maxSeqLength, paragraphStride, maxQueryLength)

		trainFeatures = None
		try:
			with open(cachedTrainFeaturesFile, "rb") as reader:
				trainFeatures = pickle.load(reader)
		except:
			print("Building train features...")
			trainFeatures = featurizeExamples(trainExamples, tokenizer, maxSeqLength, paragraphStride, maxQueryLength, True)
			with open(cachedTrainFeaturesFile, "wb") as writer:
				pickle.dump(trainFeatures, writer)

		if useDebug:
			with open(outputDir + "/debug/trainFeaturesDebug.json", "w") as file:
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
		trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=trainBatchSize)

	if doPredict:
		with open(outputDir + "/predictParams.txt", "w") as file:
			print(args, file=file)

		print("Starting {}dev dataset creation...".format("train " if useTrainDev else ""))
		evalExamples = readSQuADDataset(predictFile, False, squadV2=useVer2)

		if useDebug:
			with open(outputDir + "/debug/evalExamplesDebug{}.json".format("_trainDev" if useTrainDev else ""), "w") as file:
				print(json.dumps([t._asdict() for t in evalExamples], indent=2), file=file)

		cachedEvalFeaturesFile = outputDir + "/evalFeatures{}_{}_{}_{}_{}_{}.bin".format("_trainDev" if useTrainDev else "", "uncased" if doLowercase else "cased", hiddenSize, maxSeqLength, paragraphStride, maxQueryLength)

		evalFeatures = None
		try:
			with open(cachedEvalFeaturesFile, "rb") as reader:
				evalFeatures = pickle.load(reader)
		except:
			print("Building {}dev features...".format("train " if useTrainDev else ""))
			evalFeatures = featurizeExamples(evalExamples, tokenizer, maxSeqLength, paragraphStride, maxQueryLength, False)
			with open(cachedEvalFeaturesFile, "wb") as writer:
				pickle.dump(evalFeatures, writer)

		if useDebug:
			with open(outputDir + "/debug/evalFeaturesDebug{}.json".format("_trainDev" if useTrainDev else ""), "w") as file:
				print(json.dumps([t._asdict() for t in evalFeatures], indent=2), file=file)

		allInputIDs = torch.tensor([f.inputIDs for f in evalFeatures], dtype=torch.long)
		allInputMask = torch.tensor([f.inputMask for f in evalFeatures], dtype=torch.long)
		allSegmentIDs = torch.tensor([f.segmentIDs for f in evalFeatures], dtype=torch.long)
		allExampleIndex = torch.arange(allInputIDs.size(0), dtype=torch.long)
	#		allIsImpossible = torch.tensor([f.isImpossible for f in evalFeatures], dtype=torch.long)
		evalData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allExampleIndex)

		evalSampler = SequentialSampler(evalData)
		evalDataLoader = DataLoader(evalData, sampler=evalSampler, batch_size=predictBatchSize)

	if doTrain:
		if not bertTrainable:
			noFineTuningLayers = [model.module.bert] if nGPU > 1 else [model.bert]

			for l in noFineTuningLayers:
				for v in l.parameters():
					v.requires_grad = False


		print("Training...")
		model.train()

		for epoch in trange(int(trainEpochs), desc="Epoch"):
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
				with open(outputDir + "/losstrace.txt", arg) as file:
					print(loss.item(), file=file)

				optimizer.step()
				optimizer.zero_grad()

			arg = "a"
			if epoch == 0:
				arg = "w"
			with open(outputDir + "/lossEpochsTrace.txt", arg) as file:
				print(torch.mean(torch.tensor(epochLosses)).item(), file=file)

		modelToSave = model.module if hasattr(model, "module") else model

		print("Saving model...")
		outputModelFile = outputDir + "/QABERTTrained_{}_{}_{}_{}_{}_{}_{}.bin".format("uncased" if doLowercase else "cased", hiddenSize, maxSeqLength, paragraphStride, maxQueryLength, trainBatchSize, trainEpochs)
		torch.save(modelToSave.state_dict(), outputModelFile)

		print("Loading finetuned model...")
		model = models[model].loadPretrained(outputModelFile, False, "", hiddenSize)
		#model = QABERT.loadPretrained(outputModelFile, False, "", hiddenSize, shapes=linearShapes, activationFun=activationFun)


	if doPredict:
		print("Predicting...")
		model.to(device)
		model.eval()

		allResults = []
		for step, batch in enumerate(tqdm(evalDataLoader, desc="{} Batches".format("Train Dev" if useTrainDev else "Dev"))):

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

			outputPredFile = os.path.join(outputDir, "predictions{}.json".format("_trainDev" if useTrainDev else ""))
			outputNBestFile = os.path.join(outputDir, "nbest_predictions{}.json".format("_trainDev" if useTrainDev else ""))
			outputNullLogOddsFile = os.path.join(outputDir, "null_odds{}.json".format("_trainDev" if useTrainDev else ""))

		print("Writing predictions for {}dev dataset...".format("train " if useTrainDev else ""))
		writePredictions(evalExamples, evalFeatures, allResults, nBestSize, maxAnswerLength, doLowercase, outputPredFile, outputNBestFile, outputNullLogOddsFile, useVer2, 0.0)


if __name__ == "__main__":
	main()

