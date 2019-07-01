#!/usr/bin/env python3

import torch
from BERT import QABERTDebug, QABERT2LGELU, QABERT2LTanh, QABERT4L400Tanh, QABERT4L1024Tanh, QABERT4LReLU, QABERTVanilla, QABERTFail
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, type=str)
parser.add_argument("--modelName", default=None, type=str)
args = parser.parse_args()

funDict = {
	"QABERTDebug":QABERTDebug,
	"QABERT2LGELU":QABERT2LGELU,
	"QABERT2LTanh":QABERT2LTanh,
	"QABERT4L400Tanh":QABERT4L400Tanh,
	"QABERT4L1024Tanh":QABERT4L1024Tanh,
	"QABERT4LReLU":QABERT4LReLU, 
	"QABERTVanilla":QABERTVanilla, 
	"QABERTFail":QABERTFail
}

model = funDict[args.modelName].loadPretrained(args.model, False, "", 768)
model.to("cuda" if torch.cuda.is_available() else "cpu")

print(model)
