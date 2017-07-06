# LSTM-NER

## Introduction

An implementation of LSTM named entity recognition based on Keras. Using two kinds of embeddings as a representation of characters, they are char-embeddings and char-postion-embeddings.

>  Inspired by the work of Nanyun Peng and Mark Dredze. The idea of using different kinds of embeddings in a NER task is very brilliant.
>
>  And I used the same embeddings provided by their open source [repo](https://github.com/hltcoe/golden-horse).

### Reference

 **Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings**

>  Nanyun Peng and Mark Dredze 
>
>  *Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2015 
>
>  If you use the code, please kindly cite the following bibtex:
>
>  @inproceedings{peng2015ner, 
>  title={Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings.}, 
>  author={Peng, Nanyun and Dredze, Mark}, 
>  booktitle={EMNLP}, 
>  pages={548–-554}, 
>  year={2015} 
>  }

## Usage

### Some Constants

In **./keras_src/constants.py** file, I defined some constants. 

```python
# Those are some IO files' dirs
# you need change the BASE_DIR on your own PC
BASE_DIR = r'/Users/heshenghuan/Projects/lstm-ner/'
MODEL_DIR = BASE_DIR + r'models/'
DATA_DIR = BASE_DIR + r'data/'
EMBEDDING_DIR = BASE_DIR + r'embeddings/'
OUTPUT_DIR = BASE_DIR + r'export/'
```

### Training

Just run the **./main.py** file. Or specify some arguments if you need, like this:

```shell
python main.py --lr 0.005 --fine_tuning False --l2_reg 0.0002
```

Then the model will run on lr=0.005, not fine-tuning, l2_reg=0.0002 and all others default. Using `-h` will print all help informations. Some arguments are not useable now, but I will fix it as soon as possible.

```shell
python main.py -h
Using TensorFlow backend.
usage: main.py [-h] [--train_data TRAIN_DATA] [--test_data TEST_DATA]
               [--valid_data VALID_DATA] [--log_dir LOG_DIR]
               [--model_dir MODEL_DIR] [--restore_model RESTORE_MODEL]
               [--emb_type EMB_TYPE] [--emb_file EMB_FILE] [--emb_dim EMB_DIM]
               [--output_dir OUTPUT_DIR]
               [--ner_feature_thresh NER_FEATURE_THRESH] [--lr LR]
               [--keep_prob KEEP_PROB] [--fine_tuning [FINE_TUNING]]
               [--nofine_tuning] [--eval_test [EVAL_TEST]] [--noeval_test]
               [--test_anno [TEST_ANNO]] [--notest_anno] [--max_len MAX_LEN]
               [--nb_classes NB_CLASSES] [--hidden_dim HIDDEN_DIM]
               [--batch_size BATCH_SIZE] [--train_steps TRAIN_STEPS]
               [--display_step DISPLAY_STEP] [--l2_reg L2_REG]

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Training data file
  --test_data TEST_DATA
                        Test data file
  --valid_data VALID_DATA
                        Validation data file
  --log_dir LOG_DIR     The log dir
  --model_dir MODEL_DIR
                        Models dir
  --restore_model RESTORE_MODEL
                        Path of the model to restored
  --emb_type EMB_TYPE   Embeddings type: char/charpos
  --emb_file EMB_FILE   Embeddings file
  --emb_dim EMB_DIM     embedding size
  --output_dir OUTPUT_DIR
                        Output dir
  --ner_feature_thresh NER_FEATURE_THRESH
                        The minimum count OOV threshold for NER
  --lr LR               learning rate
  --keep_prob KEEP_PROB
                        dropout rate of hidden layer
  --fine_tuning [FINE_TUNING]
                        Whether fine-tuning the embeddings
  --nofine_tuning
  --eval_test [EVAL_TEST]
                        Whether evaluate the test data.
  --noeval_test
  --test_anno [TEST_ANNO]
                        Whether the test data is labeled.
  --notest_anno
  --max_len MAX_LEN     max num of tokens per query
  --nb_classes NB_CLASSES
                        Tagset size
  --hidden_dim HIDDEN_DIM
                        hidden unit number
  --batch_size BATCH_SIZE
                        num example per mini batch
  --train_steps TRAIN_STEPS
                        trainning steps
  --display_step DISPLAY_STEP
                        number of test display step
  --l2_reg L2_REG       L2 regularization weight
```





## Dependencies:
This is a Keras implementation; it requires installation of these python modules:  

1. Keras
2. Theano or Tensorflow as backend of Keras
3. jieba (a Chinese word segmentor)  

Both of them can be simply installed by `pip install {moduleName}`.

## Embeddings

1. **char-embeddings**

   Embeddings learnt from each character in a large unlabled text.

2. **char-postion embeddings**

   Character embeddings cannot distinguish between uses of the same character in different contexts, whereas word embeddings fail to make use of characters or character *n-gram*s that are part of many words.

   **'char-postion embeddings'** is a compromise to use character embeddings that are sensitive to the character's position in the word.

All of those embeddings are trained on a large corpus of Weibo messages.

## History

- **2017-07-06 ver 0.1.3**
  - Add new method 'accuracy', which used to calculate correct labels
  - Arguments 'emb\_type' & 'emb\_dir' now are deprecated.
  - New argument 'emb_file'
- **2017-04-11 ver 0.1.2**
  - Rewrite neural_tagger class method: loss.
  - Add a new tagger based Bi-LSTM + CNNs, where CNN used to extract bigram features.
- **2017-04-08 ver 0.1.1**
  - Rewrite class lstm-ner & bi-lstm-ner.
- **2017-03-03 ver 0.1.0**
  - Using tensorflow to implement the LSTM-NER model.
  - Basical function finished.
- **2017-02-26 ver 0.0.3**
  - lstm_ner basically completed.
  - viterbi decoding algorithm and sequence labeling.
  - Pretreatment completed.
- **2017-02-21 ver 0.0.2**
  - Basical structure of project
  - Added 3 module file: *features*, *pretreatment* and *constant*
  - Pretreatment's create dictionary function completed
- **2017-02-20 ver 0.0.1**
  - Initialization of this project. 
  - README file
  - Some util functions and basical structure of project