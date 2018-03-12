# Sentence summarization system with Bi-encoder-decoder LSTM model

We publish the source code of the paper [Deletion-based sentence compression using Bi-enc-dec LSTM](https://www.researchgate.net/publication/319186302_Deletion-based_sentence_compression_using_Bi-enc-dec_LSTM). If you find any trouble, please raise an issue or contact us at ``vietld@jaist.ac.jp`` (main developer) or ``nguyenml@jaist.ac.jp``.

We built a web-based application and API from this model at [our own server](https://s242-097.jaist.ac.jp/sum/en/) for English and Vietnamese. Please feel free to use for non-commercial purpose.

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

## Preparing the data
Our system use CoNLL data format. We provide the 10.000 original-compressed pairs dataset provided by Katja Filippova in the [``data``](https://github.com/laiviet/SentSum/tree/master/data) directory. 

For vietnamese dataset, we can not publish it here. If you want to use it, contact us at ``nguyenml@jaist.ac.jp``.

If you want to use your own data, please convert yours into CoNLL format and config it in ``endata.py`` file.


## Training 
```
python run.py -train -i run1 
```
