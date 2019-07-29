# QABERT
BERT implementation for question answering

# USE 
There are two ways to use our program, you can choose between a command line usage and use the program as a library.
##### Client usage:
Following an example of command line execution:
```
./trainDebug.py --outputDir ~/results/base_uncased_QABERT2LReLUSkip_bertAdam/3epochs_v1_bert_512seq_96_query --vocabFile ~/bert_pretrained_tensorflow/base_uncased/vocab.txt --modelWeights ~/bert_pretrained_tensorflow/base_uncased/bert_model.ckpt --trainFile ~/squad_bak/v1.1/train-v1.1.json --predictFile ~/squad_bak/v1.1/dev-v1.1.json --useTFCheckpoint --doTrain --doPredict --trainEpochs 3.0 --trainBatchSize 16 --maxSeqLength 512 --maxQueryLength 96 --doLowercase --bertTrainable --useDebug
```
It's your free choice the destination of the output files of QABERT but the generation of the directories should be done before the execution because our program doesn't manage to create them.

##### Library usage:
There is an example file which contain a tuple with set arguments, just follow the schema in `functionUsageExample.py`.  
This script could be used as a basefile to write your own or as is to run directly training.
The creation of the directory is managed here.

##### Arguments:
###### Required
`--outputDir` The output directory where the model checkpoints and predictions will be written.  
`--vocabFile` Use the vocabolary file which refers to the pretrained BERT from Google.  
`--modelWeights` Can be used to load a Google pretrained model to start finetuning or to load a QABERT-trained checkpoint, according with `--useTFCheckpoint` argument.  

###### Optional
`--trainFile` 
`--predictFile`
`--useTFCheckpoint`
`--trainEpochs`
`--trainBatchSize`
`--predictBatchSize`
`--paragraphStride`
`--maxSeqLength`
`--maxQueryLength`
`--useVer2`
`--learningRate`
`--nBestSize`
`--maxAnswerLength`
`--doLowercase`
`--useTrainDev`
`--bertTrainable`
`--useDebug`
`--modelName`

The following two arguments are not required but at least one of them should be set, together with the according files.

`--doTrain`
`--doPredict`
