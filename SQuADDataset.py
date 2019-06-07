import json
import math
import os
from io import open
from pprint import pprint
from Tokenization import BERTTokenizer
from collection import namedtuple
from utils import stripSpacesForRebuild

def whiteSpaceCheck(char):
	if char == " " or char == "\t" or char == "\n" or ord(char) == 0x202F:
		return True
	return False

def fillWordList(text, wordList, wordOffset):
	context = text["context"]
	prevWhiteSpace = True
	for char in context:
		if whiteSpaceCheck(char):
			prevWhiteSpace = True 
		else:
			if prevWhiteSpace:
				wordList.append(char)
			else:
				wordList[-1] += char
			prevWhiteSpace = False 
		wordOffset.append(len(wordList) - 1)

def answerElements(questionAnswer, wordOffset, squadV2):
	isImpossible = False
	startP = -1
	endP = -1
	answerTextOrig = ""
	if squadV2:
		isImpossible = questionAnswer["is_impossible"]
	if not isImpossible:
		answer = questionAnswer["answers"][0]
		answerTextOrig = answer["text"]
		answerOffset = answer["answer_start"]
		startP = wordOffset[answerOffset]
		endP = wordOffset[answerOffset + len(answerTextOrig) - 1]
	return startP, endP, answerTextOrig, isImpossible
		
def readSQuADDataset(inputFile, trainingMode, squadV2=True):
	
	with open(inputFile, "r", encoding='utf-8') as SQuAD:
		squadData = json.load(SQuAD)["data"]
		
	squadExamples = []
	for data in squadData:
		for par in data["paragraphs"]:
			wordList = []
			wordOffset = []
			fillWordList(par, wordList, wordOffset)
			for questionAnswer in par["qas"]:
				questionAnswerID = questionAnswer["id"]
				questionText = questionAnswer["question"]
				startP = None
				endP = None
				answerText = None
				isImpossible = False
				sample = {"parWords":wordList,
						  "questionAnswerID":questionAnswerID,
						  "questionText":questionText,
						  "startPos":startP,
						  "endPos":endP,
						  "answerText":answerText,
						  "isImpossible":isImpossible}
				if trainingMode:
					sample["startPos"], sample["endPos"], sample["answerText"], sample["isImpossible"] = answerElements(questionAnswer, wordOffset, squadV2)
				squadExamples.append(sample)
	return squadExamples

def featurizeExamples(examples, tokenizer, maxSeqLength, docStride, maxQueryLength, trainingMode):
	uniqueID = 1000000000
	
	features = []
	for (index, example) in enumerate(examples):
		queryTokens = tokenizer.tokenize(example["questionText"])
		
		if len(queryTokens) > maxQueryLength:
			queryTokens = queryTokens[0:maxQueryLength]
		
		tokenFirstWordpieceIndex = []
		firstWordpieceTokenIndex = []
		wordpieceParagraph = []
		
		for (i, token) in enumerate(example["parWords"]):
			firstWordpieceTokenIndex.append(len(wordpieceParagraph))
			subTokens = tokenizer.tokenize(token)
			for subToken in subTokens:
				tokenFirstWordpieceIndex.append(i)
				wordpieceParagraph.append(subToken)
				
		tokStartPos = None
		tokEndPos = None 
		if trainingMode:
			if example["isImpossible"]:
				tokStartPos = -1
				tokEndPos = -1
			else:
				tokStartPos = firstWordpieceTokenIndex[exampe["startPos"]]
				if example["endPos"] < len(example["parWords"]) - 1:
					tokEndPos = firstWordpieceTokenIndex[example["endPos"] + 1] -1
				else:
					tokEndPos = len(wordpieceParagraph) - 1
				(tokStartPos, tokEndPos) = improveAnswerExtent(wordpieceParagraph, tokStartPos, tokEndPos, tokenizer, example["questionText"])
		
		maxTockensForChunks = maxSeqLength - len(queryTokens) - 3
		
		docSpan = namedTuple("docTuple", ["start", "length"])
		parChunks = [] #doc_spans
		startOffset = 0
		while startOffset < len(wordpieceParagraph):
			length = len(wordpieceParagraph) - 	startOffset
			if length > maxTockensForChunks:
				length = maxTockensForChunks
			parChunks.append(docSpan(start=startOffset, length=length))
			if startOffset + length == len(wordpieceParagraph):
				break
			startOffset += min(length, docStride)
			
		for (parChuckIndex, parChunk) in enumerate(parChunks):
			tokens = []
			tokenFirstWordpieceMap = {}
			tokenMostRelevantChunk = {}
			segmentIDs = []

			# Question tokens have ID=0 while paragraph has ID=1
			# [SEP] is used to separate question and paragraph chunk
			tokens.append("[CLS]")
			segmentIDs.append(0)

			for token in queryTokens:
				tokens.append(token)
				segmentIDs.append(0)

			tokens.append("[SEP]")
			segmentIDs.append(0)

			for i in range(parChunk.length):
				splitTokenIndex = parChunk.start + i
				tokenFirstWordpieceMap[len(tokens)] = tokenFirstWordpieceIndex[splitTokenIndex]
				isMostRelevant = isMostRelevantParagraphChunk(parChunks, parChunkIndex, splitTokenIndex)
				tokenMostRelevantChunk[len(tokens)] = isMostRelevant
				tokens.append(wordpieceParagraph[splitTokenIndex])
				segmentIDs.append(1)

			tokens.append("[SEP]")
			segmentIDs.append(1)

			bertInputIDs = tokenizer.convertToIDs(tokens)
			bertInputMask = [1] * len(bertInputIDs)

			#Padding remaining positions
			while len(bertInputIDs) < maxSeqLength:
				bertInputIDs.append(0)
				bertInputMask.append(0)
				segmentIDs.append(0)

			assert len(bertInputIDs) == maxSeqLength
			assert len(bertInputMask) == maxSeqLength
			assert len(segmentIDs) == maxSeqLength

			startPosition = None
			endPosition = None

			if trainingMode:
				if example["isImpossible"]:
					startPosition = 0
					endPosition = 0
				else:
					chunkStart = parChunk.start
					chunkEnd = parChunk.start + parChunk.length - 1
					toBeThrown = False
					if not (tokStartPos >= chunkStart and tokEndPos <= chunkEnd):
						startPosition = 0
						endPosition = 0
					else:
						chunkOffset = len(queryTokens) + 2 # because of tags like [CLS] or [SEP]
						startPosition = tokStartPos - chunkStart + chunkOffset
						endPosition = tokEndPos - chunkStart + chunkOffset

 			InputFeatures = namedtuple("InputFeatures", ["ID", "exampleID", "chunkID", "tokens", "tokenFirstWordpieceMap", "tokenMostRelevantChunk", "inputIDs", "inputMask", "segmentIDs", "startPos", "endPos", "isImpossible"])

 			inputFeat = InputFeatures(ID=uniqueID,
									  exampleID=index,
									  chunkID=parChunkIndex,
									  tokens=tokens,
									  tokenFirstWordpieceMap=tokenFirstWordpieceMap,
									  tokenMostRelevantChunk=tokenMostRelevantChunk,
									  inputIDs=bertInputIDs,
									  inputMask=bertInputMask,
									  segmentIDs=segmentIDs,
									  startPos=startPosition,
									  endPos=endPosition,
									  isImpossible=example["isImpossible"])
			featues.append(inputFeat)
			uniqueID += 1

	return featues

def improveAnswerExtent(chunk, inputStart, inputEnd, tokenizer, originalAnswer):
	tokenizedAnswer = " ".join(tokenizer.tokenize(originalAnswer))

	for newStart in range(inputStart, inputEnd + 1):
		for newEnd in range(inputEnd, newStart - 1, -1):
			span = " ".join(chunk[newStart:(newEnd+1)])
			if span == tokenizedAnswer:
				return (newStart, newEnd)

	return (inputStart, inputEnd)
	
def isMostRelevantParagraphChunk(chunks, curChunkIndex, pos):
	bestScore = None
	bestScoreIndex = None

	for (index, chunk) in enumerate(chunks):
		end = chunk.start + chunk.length - 1
		if (pos < chunk.start) or (pos > end):
			continue

		leftContext = pos - chunk.start
		rightContext = end - pos
		score = min(leftContext, rightContext) + 0.01 * chunk.length
		if bestScore is None or score > bestScore:
			bestScore = score
			bestScoreIndex = index

	return curChunkIndex == bestScoreIndex


def rebuildOriginalText(predictedText, originalText, usingLowercase):
	tokenizer = BERTTokenizer(lowercase=usingLowercase)
	tokenizedText = tokenizer.basicTokenization(originalText)

	startPos = tokenizedText.find(predictedText)
	if startPos == -1:
		return originalText

	endPos = startPos + len(predictedText) - 1

	originalTextStrip, originalTextMap = stripSpacesForRebuild(originalText)
	tokenizedTextStrip, tokenizedTextMap = stripSpacesForRebuild(tokenizedText)

	if len(originalTextStrip) != len(tokenizedTextStrip):
		return originalText

	tokenMapping = {}
	for (i, tokenIndex) in tokenizedTextMap.items():
		tokenMapping[tokenIndex] = i

	originalStartPos = None
	if startPos in tokenMapping:
		mappedStartPosition = tokenMapping[startPos]
		if mappedStartPosition in originalTextMap:
			originalStartPos = originalTextMap[mappedStartPosition]

	if originalStartPos is None:
		return originalText

	originalEndPos = None
	if endPos in tokenMapping:
		mappedEndPosition = tokenMapping[endPos]
		if mappedEndPosition in originalTextMap:
			originalEndPos = originalTextMap[mappedEndPosition]

	if originalEndPos is None:
		return originalText

	return originalText[originalStartPos:(originalEndPos + 1)]





