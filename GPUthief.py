#!/usr/bin/env python3

from main import main, setTGBot
import argparse
from collections import namedtuple
import os
from os.path import expanduser
from pathlib import Path

def start():
	parser = argparse.ArgumentParser()
	# Required arguments
	parser.add_argument("--token", default=None, type=str, help="The output directory where the model checkpoints and predictions will be written.")
	parser.add_argument("--chatId", default=None, type=str)
	args = parser.parse_args()
	
	trainSample = namedtuple("trainSample", ["vocabFile", "modelWeights", "trainFile", "predictFile", "useTFCheckpoint", "doTrain", "doPredict", "trainEpochs", "trainBatchSize", "predictBatchSize", "paragraphStride", "maxSeqLength", "maxQueryLength", "useVer2", "learningRate", "nBestSize", "maxAnswerLength", "doLowercase", "useTrainDev", "bertTrainable", "useDebug", "linearShapes", "activationFun", "model"], defaults= ("~/bert_pretrained_tensorflow/base_uncased/vocabfile.txt", None, None, None, False, False, False, 1.0, 12, 8, 128, 384, 64, False, 5e-5, 20, 30, True, False, False, False, None, None, "QABERT2LReLUSkip"))
	
	trainTuples = [
		trainSample(vocabFile="~/bert_pretrained_tensorflow/base_uncased/vocabfile.txt", modelWeights="~/bert_pretrained_tensorflow/base_uncased/bert_model.ckpt", trainFile="~/squad_bak/v2/dev-v2.0.json", predictFile="~/squad_bak/v2/dev-v2.0.json", useTFCheckpoint=True, doTrain=True, doPredict=True, trainEpochs=1.0, trainBatchSize=16, maxSeqLength=512, maxQueryLength=96, useVer2=True, doLowercase=True, bertTrainable=True, useDebug=True, model="QABERT2LReLUSkip"),
		trainSample(vocabFile="~/bert_pretrained_tensorflow/base_uncased/vocabfile.txt", modelWeights="~/bert_pretrained_tensorflow/base_uncased/bert_model.ckpt", trainFile="~/squad_bak/v2/dev-v2.0.json", predictFile="~/squad_bak/v2/dev-v2.0.json", useTFCheckpoint=True, doTrain=True, doPredict=True, trainEpochs=1.0, trainBatchSize=16, maxSeqLength=512, maxQueryLength=96, useVer2=True, doLowercase=True, bertTrainable=True, useDebug=True, model="QABERT4LGELUSkip")
	]
	
	setTGBot(args.token, args.chatId)
	for train in trainTuples:
		balanced = False
		if train.doTrain and "balance" in train.trainFile:
			balanced = True
		path = "~/results/base_uncased_{}_bertAdam/{}epochs_{}_{}_{}_{}{}".format(train.model, int(train.trainEpochs), "v2" if train.useVer2 else "v1", train.maxSeqLength, train.maxQueryLength, "bert" if train.bertTrainable else "", "_balanced" if balanced else "")
		path=expanduser(path)
		try:
			os.makedirs(path, 0o755)
		except:
			print("already exists")
			pass
		if Path(path).exists:
			try:
				main(path, train.vocabFile, train.modelWeights, train.trainFile, train.predictFile, train.useTFCheckpoint, train.doTrain, train.doPredict, train.trainEpochs, train.trainBatchSize, train.predictBatchSize, train.paragraphStride, train.maxSeqLength, train.maxQueryLength, train.useVer2, train.learningRate, train.nBestSize, train.maxAnswerLength, train.doLowercase, train.useTrainDev, train.bertTrainable, train.useDebug, train.linearShapes, train.activationFun, train.model)
			except:
				continue
			
		
if __name__ == "__main__":
	start()