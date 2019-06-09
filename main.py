#!/usr/bin/env python3
import torch 
import random
import numpy as np
from BERT import QABERT
from SQuADDataset import featurizeExamples, readSquadDataset

def main():
	
	seed = 42
	
	trainBatchSize = 12
	numTrainEpochs = 2 # these both are parameter that have to come from the commmand line
	
	device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
	n_gpu = torch.cuda.device_count()
	random.seed(seed)
	np.random.seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)
	
	tokenizer = BERTTokenizer("./vocab.txt")
	trainExamples = readSquadDataset("./train-v2.0.json", True, squadV2=True)
	numTrainOptimizationStep = len(trainExamples) // trainBatchSize * numTrainEpochs
	model = QABERT.loadPretrained("../BERT Checkpoints PyTorch/Bert Base Uncased/bert-base-uncased.bin", False, "")
	model.to(device)
	if n_gpu > 1:
		model = torch.nn.DataParallel(model)
	
	no_decay = ['bias', 'NormLayer.bias', 'NormLayer.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	
	#TODO: optimizer has to be implemented
	
	globalStep = 0
	trainFeatures = featurizeExamples(trainExamples, tokenizer, 512, 128, 256, True)#to generalize paramenters
	
	allInputIDs = torch.tensor([f.inputIDs for f in trainFeatures], dtype=torch.long)
	allInputMask = torch.tensor([f.inputMask for f in trainFeatures], dtype=torch.long)
	allSegmentIDs = torch.tensor([f.segmentIDs for f in trainFeatures], dtype=torch.long)
	allStartPos = torch.tensor([f.startPos for f in trainFeatures], dtype=torch.long)
	allEndPos = torch.tensor([f.endPos for f in trainFeatures], dtype=torch.long)
	
	trainData = TensorDataset(allInputIDs, allInputMask, allSegmentIDs, allStartPos, allEndPos)
	trainSampler = DistributedSampler(trainData)
	trainDataLoader = DataLoader(trainData, sampler=trainSampler, batch_size=trainBatchSize)
	model.train()
	

	
	
	
	
	
	
if __name__ == "__main__":
	main()