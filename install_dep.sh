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


