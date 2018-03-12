import io, os
from os.path import join
import gensim.models.word2vec as word
import numpy as np
import pickle
PROJECT_DIR = './'


class CompressionData(object):
    d_fillipova = {}
    d_fillipova['train'] = PROJECT_DIR + 'data/dtrain/'
    d_fillipova['test'] = PROJECT_DIR + 'data/dtest/'
    d_fillipova['dev'] = PROJECT_DIR + 'data/dvalid/'
    d_fillipova['max_len'] = 192
    d_fillipova['description'] = 'src=google;' \
                               'given-author=\'Fillipova\'' \
                               'tokenized=false;' \
                               'unicode=true'

    fillipova = {}
    fillipova['train'] = PROJECT_DIR + 'data/train/'
    fillipova['test'] = PROJECT_DIR + 'data/test/'
    fillipova['dev'] = PROJECT_DIR + 'data/valid/'
    fillipova['max_len'] = 192
    fillipova['description'] = 'src=google;' \
                             'given-author=\'Fillipova\'' \
                             'tokenized=false;' \
                             'unicode=true'

    fillipova_conll = {}
    fillipova_conll['train'] = PROJECT_DIR + 'data/fillipova-train.conll'
    fillipova_conll['test'] = PROJECT_DIR + 'data/fillipova-test.conll'
    fillipova_conll['dev'] = PROJECT_DIR + 'data/fillipova-dev.conll'
    fillipova_conll['max_len'] = 192
    fillipova_conll['description'] = 'src=google;' \
                                   'given-author=\'Fillipova\'' \
                                   'tokenized=false;' \
                                   'unicode=true'
    d_fillipova_conll = {}
    d_fillipova_conll['train'] = PROJECT_DIR + 'data/d_fillipova-train.conll'
    d_fillipova_conll['test'] = PROJECT_DIR + 'data/d_fillipova-test.conll'
    d_fillipova_conll['dev'] = PROJECT_DIR + 'data/d_fillipova-dev.conll'
    d_fillipova_conll['max_len'] = 192
    d_fillipova_conll['description'] = 'src=google;' \
                                   'given-author=\'Fillipova\'' \
                                   'tokenized=false;' \
                                   'unicode=true'


def get_rare_vector(embedding_size):
    try:
        f = open('data/rare%d.data'%(embedding_size), 'rb')
        x = pickle.load(f)
        f.close()
        return x
    except:
        return np.random.uniform(0, 1e-3, size=(1, 1, embedding_size))

class Data():
    def __init__(self, config):
        self.config = config
        self.train_data_path = config.train_data_path
        self.valid_data_path = config.valid_data_path
        self.test_data_path = config.test_data_path
        self.vocab = {}
        self.out_vocab = {}

        self.vectors = word.Word2Vec.load_word2vec_format(config.vector_path, binary=True)
        self.embedding_size = len(self.vectors[self.vectors.vocab[0]])

        self.train_data = self.load_conll_data(self.train_data_path)
        self.valid_data = self.load_conll_data(self.valid_data_path)
        self.test_data = self.load_conll_data(self.test_data_path)

    def log(self, message):
        pass 

    def load_conll_data(self, path):
        matched = 0
        lower = 0
        unmatched = 0
        max_len = 0
        unk = pickle.load()

        def fit_max_len(data):
            x = [0]*len(data[0])
            for i in range(len(data), max_len):
                data.append(x)
            return np.array(data)

        tags = {}
        f = io.open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        data = []

        words = []
        stems = []
        features = []
        bin = []
        sample_count = 0
        for idx, line in enumerate(lines):
            x = line.split()
            if (len(x) < 2):
                sample_count += 1
                new_sample = {'word': word, 'stem': stem, 'bin': bin, 'name': str(sample_count)}
                data.append(new_sample)
                word = []
                stem = []
                bin = []
            elif len(x[0]) == 0 or len(x[1]) == 0:
                print('[Error] corpus error 1')
            else:
                w = x[0]
                l = x[1]
                if w in self.vectors.vocab:
                    features.append(self.vectors[w])
                    matched +=1
                elif w.lower() in self.vectors.vocab:
                    features.append(self.vectors[w.lower()])
                    lower +=1
                else:
                    features.append(get_rare_vector())
        return data
