#!/usr/bin/env python3

import argparse
import json
import pickle
from SQuADDataset import readSQuADDataset, featurizeExamples
from Tokenization import BERTTokenizer

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--inputFile", type=str, required=True)
	parser.add_argument("--outputFile", type=str, required=True)
	parser.add_argument("--vocabFile", type=str, required=True)
	args = parser.parse_args()

	examples = readSQuADDataset(args.inputFile, False, squadV2=True)

	with open(args.outputFile, "w+", encoding="utf-8") as file:
		json.dump(examples, file)

	tokenizer = BERTTokenizer(args.vocabFile, True)

	features = None
	try:
		with open(args.outputFile + ".features.bin", "rb") as file:
			features = pickle.load(file)
	except:
		print("Building features...");
		features = featurizeExamples(examples, tokenizer, 384, 128, 64, False)
		with open(args.outputFile + ".features.bin", "wb") as file:
			pickle.dump(features, file)

	with open(args.outputFile + ".features.bin.json", "w", encoding="utf-8") as file:
		print(json.dumps([t._asdict() for t in features[:1]]), file=file)


if __name__ == "__main__":
	main()
