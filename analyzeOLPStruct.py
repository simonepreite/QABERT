#!/usr/bin/env python3

import argparse
import json
import hickle
from SQuADDataset import readSQuADDataset, featurizeExamples
from Tokenization import BERTTokenizer
import multiprocessing

def multiprocessFeaturize(examples, tokenizer, maxSeqLength, maxParLength, maxQueryLength, trainingMode, filenames):

	procs = []
	for (index, file) in enumerate(filenames):
		if index == len(filenames)-1:
			examplesSlice = examples[index*64:]
		else:
			examplesSlice = examples[index*64:index*64+64]
		args = (examplesSlice, tokenizer, maxSeqLength, maxParLength, maxQueryLength, trainingMode, [filenames[index]])
		procs.append(multiprocessing.Process(target=featurizeExamples, args=args))

	for p in procs:
		p.start()

	for p in procs:
		p.join()

	print("DONE!")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--inputFile", type=str, required=True)
	parser.add_argument("--outputDir", type=str, required=True)
	parser.add_argument("--vocabFile", type=str, required=True)
	args = parser.parse_args()

	examples = readSQuADDataset(args.inputFile, False, squadV2=True)

	#with open(args.outputFile, "w+", encoding="utf-8") as file:
	#	json.dump(examples, file)

	tokenizer = BERTTokenizer(args.vocabFile, True)

	numFiles = (len(examples) // 64 + 1) if len(examples) % 64 else len(examples) // 64
	cachedEvalFeaturesFileNames = [args.outputDir + "/evalFeatures_file{}.bin".format(i) for i in range(1, numFiles+1)]
	print("examples length: {}".format(len(examples)))
	print("filenames length: {}".format(len(cachedEvalFeaturesFileNames)))

	evalFeatures = []
	try:
		#assert len(examples) == len(cachedEvalFeaturesFileNames)

		for elem in cachedEvalFeaturesFileNames:
			with open(elem, "rb") as file:
				print("Loading feature file: {}...".format(elem))
				evalFeatures.append(hickle.load(file, safe=False))

		print(evalFeatures[0])
	except:
		print("Building features...");
		multiprocessFeaturize(examples, tokenizer, 384, 128, 64, False, cachedEvalFeaturesFileNames)
		#features = featurizeExamples(examples, tokenizer, 384, 128, 64, False, cachedEvalFeaturesFileNames)

		#assert len(features) == len(cachedEvalFeaturesFileNames)

		# for (index, elem) in enumerate(cachedEvalFeaturesFileNames):
		# 	with open(elem, "wb") as file:
		# 		print("Saving feature file: {}...".format(elem))
		# 		hickle.dump(features[index], elem, compression="gzip", track_times=False)

	#with open(args.outputFile + ".features.bin.json", "w", encoding="utf-8") as file:
	#	print(json.dumps([t._asdict() for t in features[:1]]), file=file)


if __name__ == "__main__":
	main()
