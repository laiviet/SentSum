# Sentence summarization system with Bi-encoder-decoder LSTM model

We publish the source code of the paper [Deletion-based sentence compression using Bi-enc-dec LSTM](https://www.researchgate.net/publication/319186302_Deletion-based_sentence_compression_using_Bi-enc-dec_LSTM)
 
 This is a webserver of the sentence summarization system for English and Vietnamese with two model: Bi-Enc-Dec and Bi-LSTM-CRF.
 If you find any trouble, please raise an issue or contact us at vietld@jaist.ac.jp.
 
 ## Dependencies
 python=2.7 and python 3.5 (This project contains a webserver written in Python3 and a backend written in python2)
 
 Tensorflow=0.12.1
 
 NLTK=3.2.5
 
 Gensim=3.4.0
 
 Stanford Postagger 3.6
 
 ## Installation
 
 Clone the repository
 ```
 git clone git@github.com:nguyenlab/SentSum.git
 
 ```

 Install the dependencies for python 2 instance
 ```
 chmod +x install_dep.sh
 ./install_dep.sh
 ```
 
## Preparing the data set.
We follow the CoNLL format. Please convert your data into CoNLL. We provide the converted CoNLL format in the ``data`` directory.

## Training 

```
python run.py -train -i run1 
```
