import json
import math
import os
from io import open
from pprint import pprint
from Tokenization import BERTTokenizer, BasicTokenizer
from collections import namedtuple, defaultdict, OrderedDict
from utils import stripSpacesForRebuild, cleanWhitespaces


InputFeatures = namedtuple("InputFeatures", ["ID", "exampleID", "chunkID", "tokens", "tokenFirstWordpieceMap", "tokenMostRelevantChunk", "inputIDs", "inputMask", "segmentIDs", "startPos", "endPos", "isImpossible"])
SQuADExample = namedtuple("SQuADExample", ["parWords", "questionAnswerID", "questionText", "startPos", "endPos", "answerText", "isImpossible"])
DocSpan = namedtuple("DocSpan", ["start", "length"])
IntermediatePred = namedtuple("IntermediatePred", ["featureIndex", "startIndex", "endIndex", "startLogit", "endLogit"])
NBestPrediction = namedtuple("NBestPrediction", ["text", "startLogit", "endLogit"])
RawResult = namedtuple("RawResult", ["ID", "startLogits", "endLogits"])

def whiteSpaceCheck(char):
	if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
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

def answerElements(questionAnswer, wordList, wordOffset, squadV2):
	isImpossible = False
	startP = -1
	endP = -1
	answerTextOrig = ""
	skip = False
	if squadV2:
		isImpossible = questionAnswer["is_impossible"]
	if not isImpossible:
		answer = questionAnswer["answers"][0]
		answerTextOrig = answer["text"]
		answerOffset = answer["answer_start"]
		startP = wordOffset[answerOffset]
		endP = wordOffset[answerOffset + len(answerTextOrig) - 1]

		actualText = " ".join(wordList[startP:(endP + 1)])
		cleanedAnswer = " ".join(cleanWhitespaces(answerTextOrig))
		if actualText.find(cleanedAnswer) == -1:
			skip = True

	return startP, endP, answerTextOrig, isImpossible, skip
		
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
#				if trainingMode:
				startP, endP, answerText, isImpossible, skip = answerElements(questionAnswer, wordList, wordOffset, squadV2)

				if not skip:
					sample = SQuADExample(parWords=wordList, questionAnswerID=questionAnswerID, questionText=questionText, startPos=startP, endPos=endP, answerText=answerText, isImpossible=isImpossible)
					squadExamples.append(sample)
	return squadExamples

def getBestIndexes(logits, nBestSize):
	indexAndScore = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
	bestIndexes = []
	"""
	i = 0
	while i < len(indexAndScore) and i < nBestSize:
		bestIndexes.append(indexAndScore[i][0])
		i += 1
	"""
	for i in range(len(indexAndScore)):
		if i >= nBestSize:
			break
		bestIndexes.append(indexAndScore[i][0])
	return bestIndexes
		
def computeSoftmax(scores):
	if not scores:
		return []

	maxScore = None
	for score in scores:
		if maxScore is None or score > maxScore:
			maxScore = score

	expScores = []
	totalSum = 0.0
	for score in scores:
		x = math.exp(score - maxScore)
		expScores.append(x)
		totalSum += x

	probs = []
	for score in expScores:
		probs.append(score / totalSum)

	return probs
	"""
	maxScore = max(scores)
	expScores = []
	probs = []
	totSum = 0.0
	
	for score in scores:
		x = math.exp(score - maxScore)
		expScores.append(x)
		totSum += x
	
	for score in expScores:
		probs.append(score / totSum)
	return probs
	"""
	


def featurizeExamples(examples, tokenizer, maxSeqLength, docStride, maxQueryLength, trainingMode):
	uniqueID = 1000000000
	
	features = []
	for (index, example) in enumerate(examples):
		queryTokens = tokenizer.tokenize(example.questionText)
		
		if len(queryTokens) > maxQueryLength:
			queryTokens = queryTokens[0:maxQueryLength]
		
		tokenFirstWordpieceIndex = []
		firstWordpieceTokenIndex = []
		wordpieceParagraph = []
		
		for (i, token) in enumerate(example.parWords):
			firstWordpieceTokenIndex.append(len(wordpieceParagraph))
			subTokens = tokenizer.tokenize(token)
			for subToken in subTokens:
				tokenFirstWordpieceIndex.append(i)
				wordpieceParagraph.append(subToken)
				
		tokStartPos = None
		tokEndPos = None 
		if trainingMode:
			if example.isImpossible:
				tokStartPos = -1
				tokEndPos = -1
			else:
				tokStartPos = firstWordpieceTokenIndex[example.startPos]
				if example.endPos < len(example.parWords) - 1:
					tokEndPos = firstWordpieceTokenIndex[example.endPos + 1] -1
				else:
					tokEndPos = len(wordpieceParagraph) - 1
				(tokStartPos, tokEndPos) = improveAnswerExtent(wordpieceParagraph, tokStartPos, tokEndPos, tokenizer, example.answerText)
		
		maxTokensForChunks = maxSeqLength - len(queryTokens) - 3
		
		parChunks = [] #doc_spans
		startOffset = 0
		while startOffset < len(wordpieceParagraph):
			length = len(wordpieceParagraph) - 	startOffset
			if length > maxTokensForChunks:
				length = maxTokensForChunks
			parChunks.append(DocSpan(start=startOffset, length=length))
			if startOffset + length == len(wordpieceParagraph):
				break
			startOffset += min(length, docStride)
			
		for (parChunkIndex, parChunk) in enumerate(parChunks):
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

			bertInputIDs = tokenizer.tokensToIDs(tokens)
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
				if example.isImpossible:
					startPosition = 0
					endPosition = 0
				else:
					chunkStart = parChunk.start
					chunkEnd = parChunk.start + parChunk.length - 1
					if not (tokStartPos >= chunkStart and tokEndPos <= chunkEnd):
						startPosition = 0
						endPosition = 0
					else:
						chunkOffset = len(queryTokens) + 2 # because of tags like [CLS] or [SEP]
						startPosition = tokStartPos - chunkStart + chunkOffset
						endPosition = tokEndPos - chunkStart + chunkOffset

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
									  isImpossible=example.isImpossible)
			features.append(inputFeat)
			uniqueID += 1

	return features

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
	tokenizer = BasicTokenizer(doLowercase=usingLowercase)
	tokenizedText = " ".join(tokenizer.tokenize(originalText))

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

def writePredictions(all_examples, all_features, all_results, n_best_size,
					  max_answer_length, do_lower_case, output_prediction_file,
					  output_nbest_file, output_null_log_odds_file,
					  version_2_with_negative, null_score_diff_threshold):
	"""Write final predictions to the json file and log-odds of null if needed."""
	#logger.info("Writing predictions to: %s" % (output_prediction_file))
	#logger.info("Writing nbest to: %s" % (output_nbest_file))

	example_index_to_features = defaultdict(list)
	for feature in all_features:
		example_index_to_features[feature.exampleID].append(feature)

	unique_id_to_result = {}
	for result in all_results:
		unique_id_to_result[result.ID] = result

	#_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
	#	"PrelimPrediction",
	#	["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

	all_predictions = OrderedDict()
	all_nbest_json = OrderedDict()
	scores_diff_json = OrderedDict()

	for (example_index, example) in enumerate(all_examples):
		features = example_index_to_features[example_index]

		prelim_predictions = []
		# keep track of the minimum score of null start+end of position 0
		score_null = 1000000  # large and positive
		min_null_feature_index = 0  # the paragraph slice with min null score
		null_start_logit = 0  # the start logit at the slice with min null score
		null_end_logit = 0  # the end logit at the slice with min null score
		for (feature_index, feature) in enumerate(features):
			result = unique_id_to_result[feature.ID]
			start_indexes = getBestIndexes(result.startLogits, n_best_size)
			end_indexes = getBestIndexes(result.endLogits, n_best_size)
			# if we could have irrelevant answers, get the min score of irrelevant
			if version_2_with_negative:
				feature_null_score = result.startLogits[0] + result.endLogits[0]
				if feature_null_score < score_null:
					score_null = feature_null_score
					min_null_feature_index = feature_index
					null_start_logit = result.startLogits[0]
					null_end_logit = result.endLogits[0]
			for start_index in start_indexes:
				for end_index in end_indexes:
					# We could hypothetically create invalid predictions, e.g., predict
					# that the start of the span is in the question. We throw out all
					# invalid predictions.
					if start_index >= len(feature.tokens):
						continue
					if end_index >= len(feature.tokens):
						continue
					if start_index not in feature.tokenFirstWordpieceMap:
						continue
					if end_index not in feature.tokenFirstWordpieceMap:
						continue
					if not feature.tokenMostRelevantChunk.get(start_index, False):
						continue
					if end_index < start_index:
						continue
					length = end_index - start_index + 1
					if length > max_answer_length:
						continue
					prelim_predictions.append(
						IntermediatePred(
							featureIndex=feature_index,
							startIndex=start_index,
							endIndex=end_index,
							startLogit=result.startLogits[start_index],
							endLogit=result.endLogits[end_index]))
		if version_2_with_negative:
			prelim_predictions.append(
				IntermediatePred(
					featureIndex=min_null_feature_index,
					startIndex=0,
					endIndex=0,
					startLogit=null_start_logit,
					endLogit=null_end_logit))
		prelim_predictions = sorted(
			prelim_predictions,
			key=lambda x: (x.startLogit + x.endLogit),
			reverse=True)

		#_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name "NbestPrediction", ["text", "start_logit", "end_logit"])

		seen_predictions = {}
		nbest = []
		for pred in prelim_predictions:
			if len(nbest) >= n_best_size:
				break
			feature = features[pred.featureIndex]
			if pred.startIndex > 0:  # this is a non-null prediction
				tok_tokens = feature.tokens[pred.startIndex:(pred.endIndex + 1)]
				orig_doc_start = feature.tokenFirstWordpieceMap[pred.startIndex]
				orig_doc_end = feature.tokenFirstWordpieceMap[pred.endIndex]
				orig_tokens = example.parWords[orig_doc_start:(orig_doc_end + 1)]
				tok_text = " ".join(tok_tokens)

				# De-tokenize WordPieces that have been split off.
				tok_text = tok_text.replace(" ##", "")
				tok_text = tok_text.replace("##", "")

				# Clean whitespace
				tok_text = tok_text.strip()
				tok_text = " ".join(tok_text.split())
				orig_text = " ".join(orig_tokens)

				final_text = rebuildOriginalText(tok_text, orig_text, do_lower_case)
				if final_text in seen_predictions:
					continue

				seen_predictions[final_text] = True
			else:
				final_text = ""
				seen_predictions[final_text] = True

			nbest.append(
				NBestPrediction(
					text=final_text,
					startLogit=pred.startLogit,
					endLogit=pred.endLogit))
		# if we didn't include the empty option in the n-best, include it
		if version_2_with_negative:
			if "" not in seen_predictions:
				nbest.append(
					NBestPrediction(
						text="",
						startLogit=null_start_logit,
						endLogit=null_end_logit))
				
			# In very rare edge cases we could only have single null prediction.
			# So we just create a nonce prediction in this case to avoid failure.
			if len(nbest)==1:
				nbest.insert(0,
					NBestPrediction(text="empty", startLogit=0.0, endLogit=0.0))
				
		# In very rare edge cases we could have no valid predictions. So we
		# just create a nonce prediction in this case to avoid failure.
		if not nbest:
			nbest.append(
				NBestPrediction(text="empty", startLogit=0.0, endLogit=0.0))

		assert len(nbest) >= 1

		total_scores = []
		best_non_null_entry = None
		for entry in nbest:
			total_scores.append(entry.startLogit + entry.endLogit)
			if not best_non_null_entry:
				if entry.text:
					best_non_null_entry = entry

		probs = computeSoftmax(total_scores)

		nbest_json = []
		for (i, entry) in enumerate(nbest):
			output = OrderedDict()
			output["text"] = entry.text
			output["probability"] = probs[i]
			output["start_logit"] = entry.startLogit
			output["end_logit"] = entry.endLogit
			nbest_json.append(output)

		assert len(nbest_json) >= 1

		if not version_2_with_negative:
			all_predictions[example.questionAnswerID] = nbest_json[0]["text"]
		else:
			# predict "" iff the null score - the score of best non-null > threshold
			score_diff = score_null - best_non_null_entry.startLogit - (
				best_non_null_entry.endLogit)
			scores_diff_json[example.questionAnswerID] = score_diff
			if score_diff > null_score_diff_threshold:
				all_predictions[example.questionAnswerID] = ""
			else:
				all_predictions[example.questionAnswerID] = best_non_null_entry.text
			all_nbest_json[example.questionAnswerID] = nbest_json

	with open(output_prediction_file, "w") as writer:
		writer.write(json.dumps(all_predictions, indent=4) + "\n")

	with open(output_nbest_file, "w") as writer:
		writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

	if version_2_with_negative:
		with open(output_null_log_odds_file, "w") as writer:
			writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


"""
def writePredictions(squadExamples, squadFeatures, squadResults, nBestSize, maxAnswerLength, usingLowercase, outputPredictionPath, outputNBestPath, outputOddsPath, usingV2, nullDiffThreshold):

	print("Writing predictions to: {}".format(outputPredictionPath))
	print("Writing nbest to: {}".format(outputNBestPath))
	print("Writing (eventual) odds to: {}".format(outputOddsPath))

	exampleToFeatures = defaultdict(list)
	for f in squadFeatures:
		exampleToFeatures[f.exampleID].append(f)

	uniqueIDToResult = {}
	for res in squadResults:
		uniqueIDToResult[res.ID] = res

	allPredictions = OrderedDict()
	allNBestPredictions = OrderedDict()
	scoreDiffs = OrderedDict()

	for (exampleIndex, example) in enumerate(squadExamples):
		features = exampleToFeatures[exampleIndex]
		intermediatePreds = []

		nullScore = 1000000
		minNullFeatureIndex = 0
		nullStartLogit = 0
		nullEndLogit = 0

		for (featureIndex, feature) in enumerate(features):
			result = uniqueIDToResult[feature.ID]
			startIndexes = getBestIndexes(result.startLogits, nBestSize)
			endIndexes = getBestIndexes(result.endLogits, nBestSize)

			if usingV2:
				featureNullScore = result.startLogits[0] + result.endLogits[0]
				if featureNullScore < nullScore:
					nullScore = featureNullScore
					minNullFeatureIndex = featureIndex
					nullStartLogit = result.startLogits[0]
					nullEndLogit = result.endLogits[0]

			for startIndex in startIndexes:
				for endIndex in endIndexes:
					if (startIndex >= len(feature.tokens)) or (endIndex >= len(feature.tokens)) or (startIndex not in feature.tokenFirstWordpieceMap) or (not feature.tokenMostRelevantChunk.get(startIndex, False)) or (endIndex < startIndex) or (endIndex - startIndex + 1 > maxAnswerLength):
						continue

					ip = IntermediatePred(featureIndex=featureIndex, startIndex=startIndex, endIndex=endIndex, startLogit=result.startLogits[startIndex], endLogit=result.endLogits[endIndex])
					intermediatePreds.append(ip)

		if usingV2:
			ip = IntermediatePred(featureIndex=minNullFeatureIndex, startIndex=0, endIndex=0, startLogit=nullStartLogit, endLogit=nullEndLogit)
			intermediatePreds.append(ip)

		intermediatePreds = sorted(intermediatePreds, key=lambda x: (x.startLogit + x.endLogit), reverse=True)

		seenPredictions = {}
		nbest = []

		for pred in intermediatePreds:
			if len(nbest) >= nBestSize:
				break

			feature = features[pred.featureIndex]
			if pred.startIndex > 0:
				sIndex = pred.startIndex
				eIndex = pred.endIndex
				tokenizedTokens = feature.tokens[sIndex:(eIndex + 1)]
				originalChunkStart = feature.tokenFirstWordpieceMap[sIndex]
				originalChunkEnd = feature.tokenFirstWordpieceMap[eIndex]
				originalTokens = example.parWords[originalChunkStart:(originalChunkEnd + 1)]
				tokenizedText = " ".join(tokenizedTokens)

				tokenizedText = tokenizedText.replace(" ##", "").replace("##", "")
				tokenizedText = tokenizedText.strip()
				tokenizedText = " ".join(tokenizedText.split())
				originalText = " ".join(originalTokens)

				finalText = rebuildOriginalText(tokenizedText, originalText, usingLowercase)

				if finalText not in seenPredictions:
					seenPredictions[finalText] = True
			else:
				finalText = ""
				seenPredictions[finalText] = True

			nbestPred = NBestPrediction(text=finalText, startLogit=pred.startLogit, endLogit=pred.endLogit)
			nbest.append(nbestPred)

		if usingV2:
			if "" not in seenPredictions:
				emptyNBestPred = NBestPrediction(text="", startLogit=nullStartLogit, endLogit=nullEndLogit)
				nbest.append(emptyNBestPred)

			if len(nbest) == 1:
				nbest.insert(0, NBestPrediction(text="empty", startLogit=0.0, endLogit=0.0))

		if not nbest:
			nbest.append(NBestPrediction(text="empty", startLogit=0.0, endLogit=0.0))

		assert len(nbest) >= 1

		totalScores = []
		bestNonNullEntry = None

		for entry in nbest:
			totalScores.append(entry.startLogit + entry.endLogit)
			if (not bestNonNullEntry) and entry.text:
				bestNonNullEntry = entry

		probs = computeSoftmax(totalScores)

		nbestJSON = []
		for (i, entry) in enumerate(nbest):
			output = OrderedDict()
			output["text"] = entry.text
			output["probability"] = probs[i]
			output["start_logit"] = entry.startLogit
			output["end_logit"] = entry.endLogit
			nbestJSON.append(output)

		assert len(nbestJSON) >= 1

		if not usingV2:
			allPredictions[example.questionAnswerID] = nbestJSON[0]["text"]
		else:
			diff = nullScore - bestNonNullEntry.startLogit - bestNonNullEntry.endLogit
			scoreDiffs[example.questionAnswerID] = diff

			if diff > nullDiffThreshold:
				allPredictions[example.questionAnswerID] = ""
			else:
				allPredictions[example.questionAnswerID] = bestNonNullEntry.text

			allNBestPredictions[example.questionAnswerID] = nbestJSON

	with open(outputPredictionPath, "w") as writer:
		writer.write(json.dumps(allPredictions, indent=4) + "\n")

	with open(outputNBestPath, "w") as writer:
		writer.write(json.dumps(allNBestPredictions, indent=4) + "\n")

	if usingV2:
		with open(outputOddsPath, "w") as writer:
			writer.write(json.dumps(scoreDiffs, indent=4) + "\n")
"""
"""
examples = readSQuADDataset("./train-v2.0.json", True)[:2]
print(examples)

tokenizer = BERTTokenizer("./vocab.txt")

features = featurizeExamples(examples, tokenizer, 384, 128, 64, True)
print(features)
"""
"""
features = []
for (example_index, example) in enumerate(examples):
	query_tokens = tokenizer.tokenize(example["questionText"])

	if len(query_tokens) > 100:
		query_tokens = query_tokens[0:max_query_length]

	tok_to_orig_index = []
	orig_to_tok_index = []
	all_doc_tokens = []
	# print(example["parWords"])
	for (i, token) in enumerate(example["parWords"]):
		orig_to_tok_index.append(len(all_doc_tokens))
		sub_tokens = tokenizer.tokenize(token)
		for sub_token in sub_tokens:
			tok_to_orig_index.append(i)
			all_doc_tokens.append(sub_token)
	print("fine")
"""








