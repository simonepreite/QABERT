#!/usr/bin/env python3

import os
from io import open
from utils import loadVocab, cleanWhitespaces, cleanAccents, splitPunctuation
from collections import OrderedDict
import unicodedata


class BERTTokenizer(object):
	def __init__(self, vocabFile, doLowercase=True, basicTokenization=True, neverSplit=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
		if not os.path.isfile(vocabFile):
			raise ValueError("Can't find a vocabulary file at path {}.".format(vocabFile))

		self.vocab = loadVocab(vocabFile)
		self.IDsToTokens = OrderedDict([(ids, token) for token, ids in self.vocab.items()])
		self.basicTokenization = basicTokenization

		if self.basicTokenization:
			self.basicTokenizer = BasicTokenizer(doLowercase=doLowercase, neverSplit=neverSplit)

		self.wordpieceTokenizer = WordpieceTokenizer(vocab=self.vocab)

	def tokenize(self, text):
		splitTokens = []
		
		if self.basicTokenization:
			for token in self.basicTokenizer.tokenize(text):
				for subToken in self.wordpieceTokenizer.tokenize(token):
					splitTokens.append(subToken)
		else:
			splitTokens = self.wordpieceTokenizer.tokenize(text)

		return splitTokens

	def tokensToIDs(self, tokens):
		IDs = []
		for token in tokens:
			IDs.append(self.vocab[token])
		return IDs

	def IDsToTokens(self, IDs):
		tokens = []
		for i in IDs:
			tokens.append(self.IDsToTokens[i])
		return tokens
	"""
	def __init__(self, vocabFile=None, lowercase=True, unsplittable=None):
		self.usingLowercase = lowercase
		self.unsplittable = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
		if unsplittable:
			self.unsplittable = self.unsplittable + unsplittable
		self.vocab = None
		if vocabFile is not None:
			self.vocab = loadVocab(vocabFile)
			self.idsToTokens = OrderedDict([(id, tok) for tok, id in self.vocab.items()])

	def tokenize(self, text):
		if self.vocab is None:
			raise AttributeError("Tokenization method requires a vocabulary")

		split = []
		basicTokenized = self.basicTokenization(text)
		#print("Basic tokenization result:", basicTokenized)
		for token in basicTokenized:
			wordpieceTokenized = self.wordpieceTokenization(token)
			for subToken in wordpieceTokenized:
				split.append(subToken)
		return split

	def convertToIDs(self, tokens):
		if self.vocab is None:
			raise AttributeError("convertToIDs method requires a vocabulary")

		ids = []
		for token in tokens:
			ids.append(self.vocab[token])
		return ids

	def convertToTokens(self, ids):
		if self.vocab is None:
			raise AttributeError("convertToTokens method requires a vocabulary")

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
		if self.vocab is None:
			raise AttributeError("wordpieceTokenization method requires a vocabulary")

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
	"""

class BasicTokenizer(object):
	def __init__(self, doLowercase=True, neverSplit=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
		self.doLowercase = doLowercase
		self.neverSplit = neverSplit

	def tokenize(self, text):
		text = self.cleanText(text)
		text = self.handleChinese(text)
		originalTokens = cleanWhitespaces(text)
		splitTokens = []

		for token in originalTokens:
			if self.doLowercase and token not in self.neverSplit:
				token = token.lower()
				token = self.stripAccents(token)
			splitTokens.extend(self.splitPunctuation(token))

		return cleanWhitespaces(" ".join(splitTokens))

	def cleanText(self, text):
		output = []
		for char in text:
			cp = ord(char)
			if cp == 0 or cp == 0xfffd or isControl(char):
				continue
			if isWhitespace(char):
				output.append(" ")
			else:
				output.append(char)

		return "".join(output)

	def stripAccents(self, text):
		text = unicodedata.normalize("NFD", text)
		output = []
		for char in text:
			cat = unicodedata.category(char)
			if cat == "Mn":
				continue
			output.append(char)

		return "".join(output)

	def splitPunctuation(self, text):
		if text in self.neverSplit:
			return [text]

		chars = list(text)
		i = 0
		newWord = True
		output = []

		while i < len(chars):
			char = chars[i]
			if isPunctuation(char):
				output.append([char])
				newWord = True
			else:
				if newWord:
					output.append([])
				newWord = False
				output[-1].append(char)
			i += 1

		return ["".join(x) for x in output]

	def handleChinese(self, text):
		output = []
		for char in text:
			cp = ord(char)
			if self.isChinese(cp):
				output.append(" ")
				output.append(char)
				output.append(" ")
			else:
				output.append(char)

		return "".join(output)

	def isChinese(self, cp):
		if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or (cp >= 0x20000 and cp <= 0x2A6DF) or (cp >= 0x2A700 and cp <= 0x2B73F) or (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F)):
			return True

		return False


class WordpieceTokenizer(object):
	def __init__(self, vocab, unknownToken="[UNK]", maxCharsPerInput=100):
		self.vocab = vocab
		self.unknownToken = unknownToken
		self.maxCharsPerInput = maxCharsPerInput

	def tokenize(self, text):
		outputTokens = []

		for token in cleanWhitespaces(text):
			chars = list(token)
			if len(chars) > self.maxCharsPerInput:
				outputTokens.append(self.unknownToken)
				continue

			isBad = False
			start = 0
			subTokens = []

			while start < len(chars):
				end = len(chars)
				curSubstring = None
				while start < end:
					substr = "".join(chars[start:end])
					if start > 0:
						substr = "##" + substr
					if substr in self.vocab:
						curSubstring = substr
						break
					end -= 1

				if curSubstring is None:
					isBad = True
					break

				subTokens.append(curSubstring)
				start = end

			if isBad:
				outputTokens.append(self.unknownToken)
			else:
				outputTokens.extend(subTokens)

		return outputTokens

# Tokenization Utility Functions
def isWhitespace(char):
	if char == " " or char == "\t" or char == "\n" or char == "\r":
		return True
	cat = unicodedata.category(char)
	if cat == "Zs":
		return True
	return False

def isControl(char):
	if char == "\t" or char == "\n" or char == "\r":
		return False
	cat = unicodedata.category(char)
	if cat.startswith("C"):
		return True
	return False

def isPunctuation(char):
	cp = ord(char)
	if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
			(cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
		return True
	cat = unicodedata.category(char)
	if cat.startswith("P"):
		return True
	return False




