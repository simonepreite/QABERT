#!/usr/bin/env python3

from collections import OrderedDict
from io import open
from itertools import chain
import os
import unicodedata

def loadVocab(vocabFilepath):
	vocab = OrderedDict()
	with open(vocabFilepath, "r", encoding="utf-8") as file:
		lines = file.readlines()
		for (index, token) in enumerate(lines):
			token = token.strip()
			vocab[token] = index
	return vocab

def saveVocab(content, vocabFilepath):
	if os.path.isdir(vocabFilepath):
		vocab = os.path.join(vocabFilepath, "vocab.txt")
	with open(vocab, "w", encoding="utf-8") as file:
		for token, index in sorted(content.items(), key=lambda x: x[1]):
			file.write(token + u'\n')
	return vocab

def cleanWhitespaces(text):
	text = text.strip()
	return text.split() if text else []

def cleanAccents(token):
	normalizedToken = unicodedata.normalize("NFD", token)
	output = []
	for char in normalizedToken:
		if unicodedata.category(char) == "Mn":
			continue
		output.append(char)
	return "".join(output)

def splitPunctuation(token):
	chars = list(token)
	newWord = True
	output = []
	puncSet = set(chain(range(33, 48), range(58, 65), range(91, 97), range(123, 127)))
	for c in chars:
		charCat = unicodedata.category(c)
		cCode = ord(c)
		if cCode in puncSet or charCat.startswith("P"):
			output.append([c])
			newWord = True
		else:
			if newWord:
				output.append([])
			newWord = False
			output[-1].append(c)
	return ["".join(c) for c in output]
