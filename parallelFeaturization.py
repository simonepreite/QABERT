#!/usr/bin/env python3

import argparse
import json
import pickle as hickle
from SQuADDataset import readSQuADDataset, parallelFeaturization
from Tokenization import BERTTokenizer
import multiprocessing as mp
import os
import itertools
#from knockknock import telegram_sender

#@telegram_sender(token="828160431:AAFaIhdaDfsOTNwV-7HQF2WEageQGuVHR7E", chat_id=15495368)
def multiprocessFeaturize(examples, tokenizer, maxSeqLength, docStride, maxQueryLength, trainingMode, chunkSize):

	cpuCount = 2 if "SLURM_JOB_CPUS_PER_NODE" not in os.environ else int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
	parallelizedData = []

	print("Spawning featurization on {} cores...".format(cpuCount))

	numSlices = len(examples) // chunkSize
	if len(examples) % chunkSize:
		numSlices +=1

	#assert numSlices == len(filenames)

	for i in range(numSlices):
		if i == numSlices-1:
			dataSlice = examples[i*chunkSize:]
		else:
			dataSlice = examples[i*chunkSize:i*chunkSize+chunkSize]

		args = (dataSlice, tokenizer, maxSeqLength, docStride, maxQueryLength, trainingMode, chunkSize, i*chunkSize)
		parallelizedData.append(args)

	pool = mp.Pool(cpuCount)
	result = pool.starmap(parallelFeaturization, parallelizedData)
	pool.close()
	pool.join()
	print("Multiprocess featurization is DONE!")
	return result


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--inputFile", type=str, required=True)
	parser.add_argument("--outputDir", type=str, required=True)
	parser.add_argument("--vocabFile", type=str, required=True)
	parser.add_argument("--examplesPerFile", type=int, default=64)
	args = parser.parse_args()

	examples = readSQuADDataset(args.inputFile, False, squadV2=True)
	tokenizer = BERTTokenizer(args.vocabFile, True)

	#numFiles = (len(examples) // args.examplesPerFile + 1) if len(examples) % args.examplesPerFile else len(examples) // args.examplesPerFile
	#cachedEvalFeaturesFileNames = [args.outputDir + "/evalFeatures_file{}.bin".format(i) for i in range(1, numFiles+1)]
	cachedEvalFeaturesFileNames = args.outputDir + "/evalFeatures_file1.bin"
	print("examples length: {}".format(len(examples)))
	print("filenames length: {}".format(len(cachedEvalFeaturesFileNames)))

	evalFeatures = []
	try:
		#for elem in cachedEvalFeaturesFileNames:
		with open(cachedEvalFeaturesFileNames, "rb") as file:
			print("Loading feature file: {}...".format(cachedEvalFeaturesFileNames))
			evalFeatures = hickle.load(file)
			print("Features loaded...")
	except:
		print("Building features...");
		evalFeatures = multiprocessFeaturize(examples, tokenizer, 384, 128, 64, False, 64)
		data = list(itertools.chain(*evalFeatures))
		with open(cachedEvalFeaturesFileNames, "wb") as file:
			hickle.dump(data, file)

		#with open(args.outputDir + "/features.txt", "w+", encoding="utf-8") as feat:
		#	print(json.dumps([t._asdict() for t in data]), file=feat)
		#print(evalFeatures[0])

if __name__ == "__main__":
	main()
