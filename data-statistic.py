import utils
from endata import CompressionData


class Statistic(object):
    def __init__(self, train_data, test_data, dev_data):
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = test_data
        self.max_len = 0
        self.n_pairs = len(train_data) + len(test_data) + len(dev_data)
        self.n_words_ori = 0.0
        self.n_words_comp = 0.0
        self.vocabs = {}
        self.train_vocabs = {}

    def analyze(self, data, train_set=False):
        for item in data:
            words = item['word']
            bin = item['bin']
            if len(words) > self.max_len:
                self.max_len = len(words)
            self.n_words_ori += len(words)
            self.n_words_comp += sum(bin)
            for idx, label in enumerate(bin):
                self.vocabs[words[idx].lower()] = 1
                if train_set:
                    self.train_vocabs[words[idx].lower()] = 1

    def run(self):
        self.analyze(self.train_data, train_set=True)
        self.analyze(self.dev_data)
        self.analyze(self.test_data)

    def show_stat(self):
        print(" ".join(['-'] * 40))
        print('Number of pairs: ', self.n_pairs)
        print('Corpus vocab:    ', len(self.vocabs))
        print('Train vocab:    ', len(self.train_vocabs))
        print('Max length:      ', self.max_len)
        print('Average length:  ', float(self.n_words_ori / self.n_pairs))
        print('Compressed rate: ', float(self.n_words_comp / self.n_words_ori))
        print('\n\nVocab coverage (train/all): ', (len(self.train_vocabs) + 1e-10)/len(self.vocabs))


mik_dataset = CompressionData.mikolov_conll
mik_train = utils.load_conll_data(mik_dataset['train'])
mik_test = utils.load_conll_data(mik_dataset['test'])
mik_dev = utils.load_conll_data(mik_dataset['dev'])
mik_data = mik_train + mik_dev + mik_test

mik_stat = Statistic(mik_train, mik_test, mik_dev)
mik_stat.run()
mik_stat.show_stat()