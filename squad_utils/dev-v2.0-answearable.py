#!/usr/bin/env python3

import os
from io import open
import json
from pprint import pprint


with open("dev-v2.0.json", "r", encoding='utf-8') as SQuAD:
	squadData = json.load(SQuAD)["data"]

outputSquad = {"version": "v2.0"}
data = list()
count = 1

for obj in squadData:
	dataObj = {}
	dataObj["title"] = obj["title"]
	pars = list()
	for par in obj["paragraphs"]:
		parObj = {}
		parObj["context"] = par["context"]
		qasList = list()
		for qas in par["qas"]:
			if not qas["is_impossible"]:
				count += 1
				qasList.append(qas)
		parObj["qas"] = qasList
		pars.append(parObj)
	dataObj["paragraphs"] = pars
	data.append(dataObj)
outputSquad["data"] = data

with open("dev-v2.0-answerable.json", "w", encoding="utf-8") as uSQuAD:
	json.dump(outputSquad, uSQuAD)

print("Answearable questions: {}\nAnswearable dataset created.".format(count))
