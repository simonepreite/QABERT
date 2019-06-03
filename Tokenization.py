#!/usr/bin/env python3

import os
from io import open
from utils import loadVocab, cleanWhitespaces, cleanAccents, splitPunctuation
from collections import OrderedDict
import unicodedata


class BERTTokenizer(object):

	def __init__(self, vocabFile, lowercase=True, unsplittable=None):
		self.usingLowercase = lowercase
		self.unsplittable = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
		if unsplittable:
			self.unsplittable = self.unsplittable + unsplittable
		self.vocab = loadVocab(vocabFile)
		self.idsToTokens = OrderedDict([(id, tok) for tok, id in self.vocab.items()])

	def tokenize(self, text):
		split = []
		basicTokenized = self.basicTokenization(text)
		print("Basic tokenization result:", basicTokenized)
		for token in basicTokenized:
			wordpieceTokenized = self.wordpieceTokenization(token)
			for subToken in wordpieceTokenized:
				split.append(subToken)
		return split

	def convertToIDs(self, tokens):
		ids = []
		for token in tokens:
			ids.append(self.vocab[token])
		return ids

	def convertToTokens(self, ids):
		tokens = []
		for index in ids:
			tokens.append(self.idsToTokens[i])
		return tokens

	# Basic Tokenization Logic
	def basicTokenization(self, text):
		cleanText = []

		# Text cleaning
		controlSet = {" ", "\t", "\n", "\r"}
		for char in text:
			c = ord(char)
			charCat = unicodedata.category(char)
			if c == 0 or c == 0xfffd or (charCat.startswith("C") and c not in controlSet):
				continue
			if c in controlSet or charCat == "Zs":
				cleanText.append(" ")
			else:
				cleanText.append(char)
		cleanText = "".join(cleanText)

		tokens = cleanWhitespaces(cleanText)
		splitTokens = []
		for tok in tokens:
			if self.usingLowercase and tok not in self.unsplittable:
				tok = tok.lower()
				tok = cleanAccents(tok)
			arg = [tok] if tok in self.unsplittable else splitPunctuation(tok)
			splitTokens.extend(arg)

		return cleanWhitespaces(" ".join(splitTokens))

	# Wordpiece Tokenization Logic
	def wordpieceTokenization(self, token):
		outputTokens = []
		for tok in cleanWhitespaces(token):
			chars = list(tok)
			if len(chars) > 100:
				outputTokens.append("[UNK]")
				continue

			badToken = False
			subTokens = []
			startIndex = 0
			while startIndex < len(chars):
				endIndex = len(chars)
				curSubStr = None
				while startIndex < endIndex:
					subStr = "".join(chars[startIndex:endIndex])
					if startIndex > 0:
						subStr = "##" + subStr
					if subStr in self.vocab:
						curSubStr = subStr
						break
					endIndex -= 1
				if curSubStr is None:
					badToken = True
					break
				subTokens.append(curSubStr)
				startIndex = endIndex

			if badToken:
				outputTokens.append("[UNK]")
			else:
				outputTokens.extend(subTokens)
		return outputTokens


