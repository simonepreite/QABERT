#!/usr/bin/env python3

import os
import pickle
import torch
import random
import numpy as np
from BERT import QABERT4LGELUSkip, QABERT2LReLUSkip, QABERT2LGELU, QABERT2LTanh, QABERT4LTanh, QABERT4LReLU, QABERTVanilla
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
from utils import saveVocab

token=""
chat_id=None

@telegram_sender(token=token, chat_id=chat_id)
def main():
	parser = argparse.ArgumentParser()

	# Required arguments
	parser.add_argument("--outputDir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
	parser.add_argument("--vocabFile", default=None, type=str, required=True)
	parser.add_argument("--modelWeights", default=None, type=str, required=True)

	# Other arguments
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
	parser.add_argument("--useTrainDev", action="store_true")
	parser.add_argument("--bertTrainable", action="store_true")
	parser.add_argument("--useDebug", action="store_true")
	parser.add_argument("--modelName", type=str, default="QABERTVanilla")
	args = parser.parse_args()

	models = {
		"QABERT4LGELUSkip":QABERT4LGELUSkip,
		"QABERT2LReLUSkip":QABERT2LReLUSkip,
		"QABERT2LGELU":QABERT2LGELU,
		"QABERT2LTanh":QABERT2LTanh,
		"QABERT4LTanh":QABERT4LTanh,
		"QABERT4LReLU":QABERT4LReLU,
		"QABERTVanilla":QABERTVanilla
	}
	
	if args.modelName not in models:
		raise Exception("Wrong model name.")

	if (not args.doTrain) and (not args.doPredict):
		raise Exception("At least one between --doTrain and --doPredict must be True.")

	if args.useDebug:
		if not os.path.isdir(args.outputDir + "/debug"):
			os.mkdir(args.outputDir + "/debug", 0o755)

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

	model =  models[args.modelName].loadPretrained(args.modelWeights, args.useTFCheckpoint, convertedWeights, hiddenSize)

	with open(args.outputDir + "/modelSummary.txt", "w") as file:
		print(model, file=file)

	model.to(device)

	if nGPU > 1:
		model = torch.nn.DataParallel(model)

	print("Starting featurization...")

	if args.doTrain:
		with open(args.outputDir + "/trainParams.txt", "w") as file:
			print(args, file=file)

		trainExamples = readSQuADDataset(args.trainFile, True, squadV2=args.useVer2)

		if args.useDebug:
			with open(args.outputDir + "/debug/trainExamplesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in trainExamples], indent=2), file=file)

		numTrainOptimizationStep = len(trainExamples) * args.trainEpochs

		paramOptimizer = list(model.named_parameters())
		noDecay = ["bias", "NormLayer.bias", "NormLayer.weight"]
		parameters = [
			{"params" : [p for n, p in paramOptimizer if not any(nd in n for nd in noDecay)], "weight_decay": 0.01},
			{"params" : [p for n, p in paramOptimizer if any(nd in n for nd in noDecay)], "weight_decay": 0.0}
		]
		optimizer = BertAdam(parameters, lr=args.learningRate, warmup=0.1, t_total=numTrainOptimizationStep)

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

		if args.useDebug:
			with open(args.outputDir + "/debug/trainFeaturesDebug.json", "w") as file:
				print(json.dumps([t._asdict() for t in trainFeatures]), file=file)

		print("Starting train dataset creation...")
		allInputIDs = torch.tensor([f.inputIDs for f in trainFeatures], dtype=torch.long)
		allInputMask = torch.tensor([f.inputMask for f in trainFeatures], dtype=torch.long)
		allSegmentIDs = torch.tensor([f.segmentIDs for f in trainFeatures], dtype=torch.long)
		allStartPos = torch.tensor([f.startPos for f in trainFeatures], dtype=torch.long)
		allEndPos = torch.tensor([f.endPos for f in trainFeatures], dtype=torch.long)

		trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos)
		trainSampler = RandomSampler(trainData)
		trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=args.trainBatchSize)

	if args.doPredict:
		with open(args.outputDir + "/predictParams.txt", "w") as file:
			print(args, file=file)

		print("Starting {}dev dataset creation...".format("train " if args.useTrainDev else ""))
		evalExamples = readSQuADDataset(args.predictFile, False, squadV2=args.useVer2)

		if args.useDebug:
			with open(args.outputDir + "/debug/evalExamplesDebug{}.json".format("_trainDev" if args.useTrainDev else ""), "w") as file:
				print(json.dumps([t._asdict() for t in evalExamples], indent=2), file=file)

		cachedEvalFeaturesFile = args.outputDir + "/evalFeatures{}_{}_{}_{}_{}_{}.bin".format("_trainDev" if args.useTrainDev else "", "uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength)

		evalFeatures = None
		try:
			with open(cachedEvalFeaturesFile, "rb") as reader:
				evalFeatures = pickle.load(reader)
		except:
			print("Building {}dev features...".format("train " if args.useTrainDev else ""))
			evalFeatures = featurizeExamples(evalExamples, tokenizer, args.maxSeqLength, args.paragraphStride, args.maxQueryLength, False)
			with open(cachedEvalFeaturesFile, "wb") as writer:
				pickle.dump(evalFeatures, writer)

		if args.useDebug:
			with open(args.outputDir + "/debug/evalFeaturesDebug{}.json".format("_trainDev" if args.useTrainDev else ""), "w") as file:
				print(json.dumps([t._asdict() for t in evalFeatures], indent=2), file=file)

		allInputIDs = torch.tensor([f.inputIDs for f in evalFeatures], dtype=torch.long)
		allInputMask = torch.tensor([f.inputMask for f in evalFeatures], dtype=torch.long)
		allSegmentIDs = torch.tensor([f.segmentIDs for f in evalFeatures], dtype=torch.long)
		allExampleIndex = torch.arange(allInputIDs.size(0), dtype=torch.long)
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

			arg = "a"
			if epoch == 0:
				arg = "w"
			with open(args.outputDir + "/lossEpochsTrace.txt", arg) as file:
				print(torch.mean(torch.tensor(epochLosses)).item(), file=file)

		modelToSave = model.module if hasattr(model, "module") else model

		print("Saving model...")
		outputModelFile = args.outputDir + "/QABERTTrained_{}_{}_{}_{}_{}_{}_{}.bin".format("uncased" if args.doLowercase else "cased", hiddenSize, args.maxSeqLength, args.paragraphStride, args.maxQueryLength, args.trainBatchSize, args.trainEpochs)
		torch.save(modelToSave.state_dict(), outputModelFile)
		saveVocab(tokenizer.vocab, args.outputDir)

		print("Loading finetuned model...")
		model = models[args.modelName].loadPretrained(outputModelFile, False, "", hiddenSize)


	if args.doPredict:
		print("Predicting...")
		model.to(device)
		model.eval()

		allResults = []
		for step, batch in enumerate(tqdm(evalDataLoader, desc="{} Batches".format("Train Dev" if args.useTrainDev else "Dev"))):

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

			outputPredFile = os.path.join(args.outputDir, "predictions{}.json".format("_trainDev" if args.useTrainDev else ""))
			outputNBestFile = os.path.join(args.outputDir, "nbest_predictions{}.json".format("_trainDev" if args.useTrainDev else ""))
			outputNullLogOddsFile = os.path.join(args.outputDir, "null_odds{}.json".format("_trainDev" if args.useTrainDev else ""))

		print("Writing predictions for {}dev dataset...".format("train " if args.useTrainDev else ""))
		writePredictions(evalExamples, evalFeatures, allResults, args.nBestSize, args.maxAnswerLength, args.doLowercase, outputPredFile, outputNBestFile, outputNullLogOddsFile, args.useVer2, 0.0)


if __name__ == "__main__":
	main()

