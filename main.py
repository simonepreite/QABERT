#!/usr/bin/env python3
import torch 
import random
import numpy as np
from BERT import QABERT
from SQuADDataset import featurizeExamples, readSquadDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

def main():
	
	seed = 42
	
	outputDir = "./results"
	trainBatchSize = 12
	numTrainEpochs = 1 # these both are parameter that have to come from the commmand line
	
	device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
	nGPU = torch.cuda.device_count()
	random.seed(seed)
	np.random.seed(seed)
	if nGPU > 0:
		torch.cuda.manual_seed_all(seed)
	
	tokenizer = BERTTokenizer("./vocab.txt")
	trainExamples = readSquadDataset("./train-v2.0.json", True, squadV2=True)
	numTrainOptimizationStep = len(trainExamples) // trainBatchSize * numTrainEpochs
	model = QABERT.loadPretrained("../BERT Checkpoints PyTorch/Bert Base Uncased/bert-base-uncased.bin", False, "")
	model.to(device)
	if nGPU > 1:
		model = torch.nn.DataParallel(model)
	
	no_decay = ['bias', 'NormLayer.bias', 'NormLayer.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	
	#TODO: optimizer has to be implemented
	optimizer = Adam(model.parameters(), lr=3e-5)
	
	globalStep = 0
	trainFeatures = featurizeExamples(trainExamples, tokenizer, 512, 128, 256, True)#to generalize paramenters
	
	allInputIDs = torch.tensor([f.inputIDs for f in trainFeatures], dtype=torch.long)
	allInputMask = torch.tensor([f.inputMask for f in trainFeatures], dtype=torch.long)
	allSegmentIDs = torch.tensor([f.segmentIDs for f in trainFeatures], dtype=torch.long)
	allStartPos = torch.tensor([f.startPos for f in trainFeatures], dtype=torch.long)
	allEndPos = torch.tensor([f.endPos for f in trainFeatures], dtype=torch.long)
	allIsImpossible = torch.torch([f.isImpossible for f in trainFeatures], dtype=torch.long)
	
	trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos, allIsImpossible)
	trainSampler = DistributedSampler(trainData)
	trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=trainBatchSize)
	model.train()

	deactivatedLayers = [model.bert, model.qaLinear1, model.qaLinear2, model.qaOutput]
	for l in deactivatedLayers:
		for v in l.paramenters():
			v.requires_grad = False
	
	# Training for the isImpossible part of the network
	for _ in range(int(numTrainEpochs)):
		for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration for IsImpossible")):
			if nGPU == 1:
				batch = tuple(t.to(device) for t in batch)

			inputIDs, inputMask, segmentIDs, startPositions, endPositions, isImpossibles = batch
			_, _, isImpossibleComputed = model(inputIDs, inputMask, segmentIDs)

			classWeights = torch.tensor([1, 2])
			weightedLossFun = CrossEntropyLoss(weight=classWeights)
			loss = weightedLossFun(isImpossibleComputed, isImpossibles)

			if nGPU > 1:
				loss = loss.mean()

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()
			globalStep += 1

	for l in deactivatedLayers:
		for v in l.paramenters():
			v.requires_grad = True

	deactivatedLayers = [model.bert, model.midIsImpossibleLinear, model.isImpossibleOutput]
	for l in deactivatedLayers:
		for v in l.paramenters():
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

	evalExamples = readSquadDataset("./dev-v2.0.json", False, squadV2=True)
	evalFeatures = featurizeExamples(evalExamples, tokenizer, 512, 128, 256, False)

	print("Predicting...")

	allInputIDs = torch.tensor([f.inputIDs for f in evalFeatures], dtype=torch.long)
	allInputMask = torch.tensor([f.inputMask for f in evalFeatures], dtype=torch.long)
	allSegmentIDs = torch.tensor([f.segmentIDs for f in evalFeatures], dtype=torch.long)
	# allStartPos = torch.tensor([f.startPos for f in evalFeatures], dtype=torch.long)
	# allEndPos = torch.tensor([f.endPos for f in evalFeatures], dtype=torch.long)
	# allIsImpossible = torch.torch([f.isImpossible for f in evalFeatures], dtype=torch.long)
	allExampleIndex = torch.arange(allInputIDs.size(0), dtype=torch.long)
	evalData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allExampleIndex)

	evalSampler = SequentialSampler(evalData)
	evalDataLoader = DataLoader(evalData, sampler=evalSampler, batch_size=trainBatchSize)

	model.eval()
	allResults = []

	for inputIDs, inputMask, segmentIDs, exampleIndices in tqdm(eval_dataloader, desc="Evaluating"):
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

		writePredictions(evalExamples, evalFeatures, allResults, nBestSize=20, maxAnswerLength=30, usingLowercase=True, outputPredFile, outputNBestFile, outputNullLogOddsFile, usingV2=True, nullDiffThreshold=0.0)

	
	
if __name__ == "__main__":
	main()