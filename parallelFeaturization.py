#!/usr/bin/env python3

import argparse
import json
import hickle
from SQuADDataset import readSQuADDataset, featurizeExamples
from Tokenization import BERTTokenizer
import multiprocessing as mp
import os

def multiprocessFeaturize(examples, tokenizer, maxSeqLength, docStride, maxQueryLength, trainingMode, chunkSize, filenames):

	cpuCount = 1 if "SLURM_JOB_CPUS_PER_NODE" not in os.environ else int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
	parallelizedData = []

	print("Spawning featurization on {} cores...".format(cpuCount))

	numSlices = len(examples) // chunkSize
	if len(examples) % chunkSize:
		numSlices +=1

	assert numSlices == len(filenames)

	for i in range(numSlices):
		if i == numSlices-1:
			dataSlice = examples[i*chunkSize:]
		else:
			dataSlice = examples[i*chunkSize:i*chunkSize+chunkSize]

		args = (dataSlice, tokenizer, maxSeqLength, docStride, maxQueryLength, trainingMode, chunkSize, [filenames[i]])
		parallelizedData.append(args)

	pool = mp.Pool(cpuCount)
	pool.starmap(featurizeExamples, parallelizedData)
	pool.close()
	pool.join()
	print("Multiprocess featurization is DONE!")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--inputFile", type=str, required=True)
	parser.add_argument("--outputDir", type=str, required=True)
	parser.add_argument("--vocabFile", type=str, required=True)
	parser.add_argument("--examplesPerFile", type=int, default=64)
	args = parser.parse_args()

	examples = readSQuADDataset(args.inputFile, False, squadV2=True)
	tokenizer = BERTTokenizer(args.vocabFile, True)

	numFiles = (len(examples) // args.examplesPerFile + 1) if len(examples) % args.examplesPerFile else len(examples) // args.examplesPerFile
	cachedEvalFeaturesFileNames = [args.outputDir + "/evalFeatures_file{}.bin".format(i) for i in range(1, numFiles+1)]
	print("examples length: {}".format(len(examples)))
	print("filenames length: {}".format(len(cachedEvalFeaturesFileNames)))

	evalFeatures = []
	try:
		for elem in cachedEvalFeaturesFileNames:
			with open(elem, "rb") as file:
				print("Loading feature file: {}...".format(elem))
				evalFeatures.append(hickle.load(file, safe=False))
	except:
		print("Building features...");
		multiprocessFeaturize(examples, tokenizer, 384, 128, 64, False, 64, cachedEvalFeaturesFileNames)


if __name__ == "__main__":
	main()
