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

waiting to be written.

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

- **2017-02-20 ver 0.0.1**
  - Initialization of this project. 
  - README file
  - Some util function​s and basical structure of project