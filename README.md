# LSTM-CRF

## Introduction

An implementation of LSTM+CRF model for Sequence labeling tasks. Based on Tensorflow(>=r1.1), and support multiple architecture like LSTM+CRF, BiLSTM+CRF, and combination of character-level CNN and BiLSTM+CRF.

Other architecture of RNN+CRF, like traditional feature involved architecture will be adding after.

## Dependecies

Because this project used Tensorflow API, it requires installation of Tensorflow and some other python modules:

- Tensorflow ( >= r1.1)

Both of them can be easily installed by `pip`.

## Data Format

The data format is basically consistent with the CRF++ toolkit. Generally speaking, training and test file must consist of multiple tokens. In addition, a token consists of multiple (but fixed-numbers) columns. Each token must be represented in one line, with the columns separated by white space (spaces or tabular characters). A sequence of token becomes a sentence.

To identify the boundary between sentences, an empty line is put. **It means there should be a '\n\n' between two different sentences.** So, if your OS is Windows, please check out what the boundary character really is.

Here's an example of such a file: (data for Chinese NER)

```text
...
感	O
动	O
了	O
李	B-PER.NAM
开	I-PER.NAM
复	I-PER.NAM
感	O
动	O

回	O
复	O
支	O
持	O
...
```

## Featrue template

This part you can read the readme file under lib directory, which is a submodule named [NeuralTextProcess](https://github.com/heshenghuan/ContextFeatureExtractor).

In file `template` specificated the feature template which used in context-based feature extraction. The second line `fields` indicates the field name for each column of a token. And the `templates` described how to extract features.

For example, the basic template is:

```text
# Fields(column), w, y, x & F are reserved names
w y
# templates.
w:-2
w:-1
w: 0
w: 1
w: 2
```

it means, each token will only has 2 columns data, 'w' and 'y'. Field `y` should always be at the last column.

> Note that `w` `y` & `F` fields are reserved, because program used them to represent word, label and word's features.
>
> Each token will become a dict type data like '{'w': '李', 'y': 'B-PER.NAM', 'F': ['w[-2]=动', 'w[-1]=了', ...]}'

The above `templates` describes a classical context feature template:

- C(n) n=-2,-1,0,1,2

'C(n)' is the value of token['w'] at relative position n.

If your token has more than 2 columns, you may need change the fields and template depends on how you want to do extraction.

In this project, I disabled prefix of feature to extract words in a context window.

## Embeddings

This program supports pretrained embeddings input. When running this program, you should give a embedding text file(word2vec tool standard output format) by specific argument.

## Usage

### Environment settings

In **env_settings.py** file, there are some environment settings like 'output dir':

```python
# Those are some IO files' dirs
# you need change the BASE_DIR on your own PC
BASE_DIR = r'project dir/'
MODEL_DIR = BASE_DIR + r'models/'
DATA_DIR = BASE_DIR + r'data/'
EMB_DIR = BASE_DIR + r'embeddings/'
OUTPUT_DIR = BASE_DIR + r'export/'
LOG_DIR = BASE_DIR + r'Summary/'
```

If your don't have those dirs in your project dir, just run `python env_settings.py`, and they will be created automatically.

### Training

#### 1. Using embeddings as features

Just run the **./main.py** file. Or specify some arguments if you need, like this:

```shell
python main.py --lr 0.005 --fine_tuning False --l2_reg 0.0002
```

Then the model will run on lr=0.005, not fine-tuning, l2_reg=0.0002 and all others default. Using `-h` will print all help informations. Some arguments are not useable now, but I will fix it as soon as possible.

```shell
python main.py -h
usage: main.py [-h] [--train_data TRAIN_DATA] [--test_data TEST_DATA]
               [--valid_data VALID_DATA] [--log_dir LOG_DIR]
               [--model_dir MODEL_DIR] [--model MODEL]
               [--restore_model RESTORE_MODEL] [--emb_file EMB_FILE]
               [--emb_dim EMB_DIM] [--output_dir OUTPUT_DIR]
               [--only_test [ONLY_TEST]] [--noonly_test] [--lr LR]
               [--dropout DROPOUT] [--fine_tuning [FINE_TUNING]]
               [--nofine_tuning] [--eval_test [EVAL_TEST]] [--noeval_test]
               [--max_len MAX_LEN] [--nb_classes NB_CLASSES]
               [--hidden_dim HIDDEN_DIM] [--batch_size BATCH_SIZE]
               [--train_steps TRAIN_STEPS] [--display_step DISPLAY_STEP]
               [--l2_reg L2_REG] [--log [LOG]] [--nolog] [--template TEMPLATE]

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
  --model MODEL         Model type: LSTM/BLSTM/CNNBLSTM
  --restore_model RESTORE_MODEL
                        Path of the model to restored
  --emb_file EMB_FILE   Embeddings file
  --emb_dim EMB_DIM     embedding size
  --output_dir OUTPUT_DIR
                        Output dir
  --only_test [ONLY_TEST]
                        Only do the test
  --noonly_test
  --lr LR               learning rate
  --dropout DROPOUT     Dropout rate of input layer
  --fine_tuning [FINE_TUNING]
                        Whether fine-tuning the embeddings
  --nofine_tuning
  --eval_test [EVAL_TEST]
                        Whether evaluate the test data.
  --noeval_test
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
  --log [LOG]           Whether to record the TensorBoard log.
  --nolog
  --template TEMPLATE   Feature templates
```

There has three type of model can be choosed by using argument '--model', they are:

1. LSTM + CRF
2. BiLSTM + CRF
3. CNN + BiLSTM + CRF

#### 2. Using both embeddings and context

We proposed a [hybrid model](./hybrid_model.py) which can use both embeddings and contextual features as input for sequence labeling task. The embeddings are used as input of RNN. And the contextual features are used like traditional feature functions in CRFs.

Just run the **./hybird_tagger.py** file. Or specify some arguments if you need, like this:

```shell
python hybrid_tagger.py -h
usage: hybrid_tagger.py [-h] [--train_data TRAIN_DATA] [--test_data TEST_DATA]
                        [--valid_data VALID_DATA] [--log_dir LOG_DIR]
                        [--model_dir MODEL_DIR]
                        [--restore_model RESTORE_MODEL] [--emb_file EMB_FILE]
                        [--emb_dim EMB_DIM] [--output_dir OUTPUT_DIR]
                        [--only_test [ONLY_TEST]] [--noonly_test] [--lr LR]
                        [--dropout DROPOUT] [--fine_tuning [FINE_TUNING]]
                        [--nofine_tuning] [--eval_test [EVAL_TEST]]
                        [--noeval_test] [--max_len MAX_LEN]
                        [--nb_classes NB_CLASSES] [--hidden_dim HIDDEN_DIM]
                        [--batch_size BATCH_SIZE] [--train_steps TRAIN_STEPS]
                        [--display_step DISPLAY_STEP] [--l2_reg L2_REG]
                        [--log [LOG]] [--nolog] [--template TEMPLATE]
                        [--window WINDOW] [--feat_thresh FEAT_THRESH]

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
  --emb_file EMB_FILE   Embeddings file
  --emb_dim EMB_DIM     embedding size
  --output_dir OUTPUT_DIR
                        Output dir
  --only_test [ONLY_TEST]
                        Only do the test
  --noonly_test
  --lr LR               learning rate
  --dropout DROPOUT     Dropout rate of input layer
  --fine_tuning [FINE_TUNING]
                        Whether fine-tuning the embeddings
  --nofine_tuning
  --eval_test [EVAL_TEST]
                        Whether evaluate the test data.
  --noeval_test
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
  --log [LOG]           Whether to record the TensorBoard log.
  --nolog
  --template TEMPLATE   Feature templates
  --window WINDOW       Window size of context
  --feat_thresh FEAT_THRESH
                        Only keep feats which occurs more than 'thresh' times.
```

### Test

If you set 'only\_test' to True or 'train\_steps' to 0, then program will only do test process.

So you must give a specific path to 'restore\_model'.

## History

- **2018-01-09 ver 0.2.4**
  - Update Neural Text Process lib 0.2.1
  - Compatible modification in main file.
- **2017-11-04 ver 0.2.3**
  - Hybrid feature architecture for LSTM and corresponding tagger's python script.
- **2017-10-31 ver 0.2.2**
  - Update Neural Text Process lib 0.2.0
  - Compatible modification in main file.
- **2017-10-20 ver 0.2.1**
  - Fix: Non-suffix for template in 'only test' process.
  - Fix: Now using correct dicts for embedding lookup table.
  - Fix: A bug of batch generator 'batch_index'.
- **2017-09-12 ver 0.2.0**
  - Update: process lib 0.1.2
  - Removed 'keras_src', completed the refactoring of the code hierarchy.
  - Added env_settings.py to make sure all default dirs exist.
  - Support restore model from file.
  - Support model selection.
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