#!/usr/bin/env python3

import os
from io import open
import json
import random
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--trainFile", default=None, required=True, type=str)
	parser.add_argument("--outputFile", default=None, required=True, type=str)
	parser.add_argument("--numSamples", default=50, required=True, type=int)
	args = parser.parse_args()

	with open(args.trainFile, "r", encoding='utf-8') as SQuAD:
		squadData = json.load(SQuAD)

	outputSquad = {"version": squadData["version"]}
	squadData = squadData["data"]

	indexList = random.sample(range(0, len(squadData)), args.numSamples)
	data = list()

	for i in indexList:
		data.append(squadData[i])

	outputSquad["data"] = data

	with open(args.outputFile, "w", encoding="utf-8") as output:
		json.dump(outputSquad, output)


if __name__ == "__main__":
	main()
