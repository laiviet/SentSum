#!/bin/bash

# Install dependency packages

pip install tensorflow-gpu=0.12
pip install gensim=3.4.0
pip install sklearn
pip install gensim
pip install nltk==3.2.5
pip install pyrouge


# Download stanford postagger and unzip
wget https://nlp.stanford.edu/software/stanford-postagger-full-2015-12-09.zip
unzip stanford-postagger-full-2015-12-09.zip


# Download word2vec (slim version)
cd data
wget https://media.githubusercontent.com/media/eyaler/word2vec-slim/master/GoogleNews-vectors-negative300-SLIM.bin.gz
echo "Extracting GoogleNews-vectors-negative300-SLIM.bin.gz"
gunzip GoogleNews-vectors-negative300-SLIM.bin.gz
cd ../


