#!/usr/bin/env python3
"""
import torch
import torch.nn as nn
from Encoder import Encoder


x = torch.FloatTensor([[[ 0.9059, -0.7039, -0.3376,  0.1968],[-1.0413,  0.8128,  0.0697, -0.6166],[-0.3793, -0.9851, -2.3841, -0.7003],[ 0.6076, -1.4874, -0.1079,  0.4266]]])

inputIDs = torch.randn(1, 4)
attentionMask = torch.ones_like(inputIDs)
extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2)
#extendedAttentionMask = extendedAttentionMask.to(dtype=next(self.parameters()).dtype)
extendedAttentionMask = (1.0 - extendedAttentionMask) * -10000.0
#tensor = torch.randn(1, 4, 4)
#print(tensor)
print()
print(extendedAttentionMask)
print()
encoder = Encoder(4, 1)
print(encoder(x, extendedAttentionMask))
"""
"""
vocabFile = "vocab.txt"
#inputSentence = "Ciao mondo, àèioù... Sticazzi!"
inputSentence = "Japanese"
#inputSentence = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."

from Tokenization import BERTTokenizer
myTokenizer = BERTTokenizer(vocabFile)
myOutput = myTokenizer.tokenize(inputSentence)
print(myOutput)
"""

from BERT import BERTModel, QABERT
from pprint import pprint

checkpointPath = "../BERT Checkpoints Tensorflow/Bert Base Uncased/bert_model.ckpt"
outputPath = "../BERT Checkpoints PyTorch/Bert Base Uncased/bert-base-uncased.bin"
# model = BERTModel.loadPretrained(outputPath, False, outputPath, 768)
qaModel = QABERT.loadPretrained(outputPath, False, outputPath, 768)

# print(qaModel.state_dict().keys())
# print("From loaded state_dict")
# print("encoder.0.0.multiHeadAtt.outputLinear.weight")
print(qaModel.state_dict()["bert.encoder.0.0.multiHeadAtt.outputLinear.weight"])



