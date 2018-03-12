# Sentence summarization system with Bi-encoder-decoder LSTM model

We publish the source code of the paper [Deletion-based sentence compression using Bi-enc-dec LSTM](https://www.researchgate.net/publication/319186302_Deletion-based_sentence_compression_using_Bi-enc-dec_LSTM). If you find any trouble, please raise an issue or contact us at vietld@jaist.ac.jp (main developer) or nguyenml@jaist.ac.jp.
 
 We built a web-based application and API from this model at [our own server](https://s242-097.jaist.ac.jp/sum/en/) for English and Vietnamese.
 
 
 
 ## Dependencies 
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
