#!/usr/bin/env python3

import torch
import torch.nn as nn
import argparse
from BERT import QABERTDebug


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--initialWeights", type=str, required=True, default=None)
	parser.add_argument("--finetunedWeights", type=str, required=True, default=None)
	args = parser.parse_args()

	print("Loading models...")
	model = QABERTDebug.loadPretrained(args.initialWeights, False, "", 768)
	ftModel = QABERTDebug.loadPretrained(args.finetunedWeights, False, "", 768)

	nGPU = torch.cuda.device_count()
	if nGPU > 1:
		model = nn.DataParallel(model)
		ftModel = nn.DataParallel(ftModel)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	ftModel.to(device)

#	bert = model.module.bert if nGPU > 1 else model.bert
#	ftBert = ftModel.module.bert if nGPU > 1 else ftModel.bert
	modelStateDict = model.state_dict()
	ftModelStateDict = ftModel.state_dict()

#	print(modelStateDict.keys())

	filtered = dict(filter(lambda x: ".bert." in x[0], modelStateDict.items()))
	filteredFT = dict(filter(lambda x: ".bert." in x[0], ftModelStateDict.items()))

	assert len(filtered) == len(filteredFT)

	for key, value in modelStateDict.items():
		print("Checking {}... {}".format(key, torch.all(torch.eq(value, ftModelStateDict[key]))))


if __name__ == "__main__":
	main()
