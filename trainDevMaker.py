#!/usr/bin/env python3

import os
from io import open
import json
import random


with open("train-v2.0.json", "r", encoding='utf-8') as SQuAD:
	squadData = json.load(SQuAD)["data"]

index_list = random.sample(range(0, len(squadData)), 50)

outputSquad = {"version": "v2.0"}
data = list()

for i in index_list:
	data.append(squadData[i])

outputSquad["data"] = data

with open("train-dev-v2.0.json", "w", encoding="utf-8") as output:
	json.dump(outputSquad, output)
	