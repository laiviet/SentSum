import pickle
import time
import os, io
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import gensim
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

import endata

log_console = True
log_file = True

tokenizer = StanfordTokenizer(
    'stanford-postagger-full-2015-12-09/stanford-postagger-3.6.0.jar')

train_path = ''
test_path = ''
word2vec_path = ''
embedding_size = 100


# tokenize the sentence by Stanford Tokenizer backend
# if format = list:    return a list of tokens
# otherwise:    return a tokenized string separated by space
def tokenize(sentence, format='list'):
    global tokenizer
    output = word_tokenize(sentence)
    if format == 'list':
        return output
    else:
        s = ''
        for i in output:
            s += i + ' '
        return s[:len(s) - 1]


def log(mess, ex_id):
    if log_console:
        print(time.ctime() + '  ' + mess)
    logfile = ex_id + '_log.txt'
    if log_file:
        try:
            f = open(logfile, 'a')
        except:
            f = open(logfile, 'w')
            f.write('Epoch: \n ')
        f.write(time.ctime() + '  ' + mess + '\n')
        f.close()


def update_epoch(epoch, sample, ex_id, test=False):
    file = ex_id + '_log.txt'
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    f = open(file, 'w')
    if test:
        f.write('Current test: ' + str(epoch) + '/' + str(sample) + '\n')
    else:
        f.write('Current train: ' + str(epoch) + '/' + str(sample) + '\n')
    for i in range(1, len(lines)):
        f.write(lines[i])
    f.close()


def load_word2vec():
    #path = 'data/onebill256.bin'
    path = 'data/GoogleNews-vectors-negative300-SLIM.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    #print('Word embedding model was loaded: ', path)
    return model


def get_word_vector(model, word, stem):
    global rare
    try:
        x = model[word]
        return np.reshape(x, [1, 1, embedding_size])
    except:
        try:
            x = model[stem]
            return np.reshape(x, [1, 1, embedding_size])
        except:
            return rare


def get_rare_vector():
    try:
        f = open('data/rare%d.data'%(embedding_size), 'rb')
        x = pickle.load(f)
        f.close()
        return x
    except:
        return np.random.uniform(0, 1e-3, size=(1, 1, embedding_size))


def get_eos_vector():
    try:
        f = open('data/eos%d.data'%(embedding_size), 'rb')
        x = pickle.load(f)
        f.close()
        return x
    except:
        return np.random.uniform(0, 1e-3, size=(1, 1, embedding_size))


def predict(output, length):
    prediction = []
    # input('Continute: ')
    for i in range(length):
        if output[i][0][0] > output[i][0][1]:
            prediction.append(0)
        else:
            prediction.append(1)
    return prediction


def load_conll_data(path):
    lemmatizer = WordNetLemmatizer()
    f = io.open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    data = []

    word = []
    stem = []
    bin = []
    sample_count = 0
    for idx, line in enumerate(lines):
        x = line.split()
        if (len(x) < 2):
            new_sample = {'word': word, 'stem': stem, 'bin': bin, 'name': str(sample_count)}
            sample_count += 1
            data.append(new_sample)
            word = []
            stem = []
            bin = []
        elif len(x[0]) == 0 or len(x[1]) == 0:
            print('[Error] corpus error 1')
        else:
            word.append(x[0])
            stem.append(lemmatizer.lemmatize(x[0]))
            if x[1] == 'B-DEL':
                bin.append(1)
            else:
                bin.append(0)
    return data


def load_conll_data_as_dict(path):
    lemmatizer = WordNetLemmatizer()
    f = io.open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    data = {}

    word = []
    stem = []
    bin = []
    sample_count = 0
    for line in lines:
        x = line.split()
        if (len(x) < 2):
            new_sample = {'word': word, 'stem': stem, 'bin': bin, 'name': str(sample_count)}
            #sample_count += 1
            data[str(sample_count)] = new_sample
            sample_count += 1
            word = []
            stem = []
            bin = []
        elif len(x[0]) == 0 or len(x[1]) == 0:
            print('[Error] corpus error')
        else:
            word.append(x[0])
            stem.append(lemmatizer.lemmatize(x[0]))
            if x[1] == 'B-DEL':
                bin.append(1)
            else:
                bin.append(0)
    return data


def load_folded_data(path):
    data = []
    files = os.listdir(path)
    for i in range(len(files)):
        f = open(path + files[i], 'rb')
        word = pickle.load(f)
        stem = pickle.load(f)
        bin = pickle.load(f)
        new_sample = {'word': word, 'stem': stem, 'bin': bin, 'name': files[i]}
        data.append(new_sample)
    return data


def load_folded_data_as_dict(path):
    data = {}
    files = os.listdir(path)
    for i in range(len(files)):
        f = open(path + files[i], 'rb')
        word = pickle.load(f)
        stem = pickle.load(f)
        bin = pickle.load(f)
        new_sample = {'word': word, 'stem': stem, 'bin': bin, 'name': files[i]}
        data[files[i]] = new_sample
    return data


def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(path, ' existed! All data will be overode.')


############

def evaluate(test_data, epoch_output_path):
    total_target = []
    total_prediction = []
    with open(epoch_output_path, 'rb') as f:
        outputs = pickle.load(f)
    for name, labels in outputs.items():
        target = test_data[name]['bin']
        prediction = predict(labels, len(target))
        total_target += target
        total_prediction += prediction
    pre_recall_fscore = precision_recall_fscore_support(total_target, total_prediction)
    acc = accuracy_score(total_target, total_prediction)
    print(pre_recall_fscore, acc)
    print('Compression rate: %f' % (sum(total_prediction) / len(total_prediction)))


def print_result(test_data, epoch_output_path):
    fname = epoch_output_path.replace('/', '-')
    f = open(fname + '.output', 'w')
    for fname in os.listdir(epoch_output_path):
        if fname.endswith('output'):
            id = fname.split('.')[0]
            item = test_data[id]
            bin = item['bin']
            ori = item['original'].split()
            length = len(bin)
            output_path = os.path.join(epoch_output_path, fname)
            outputs = pickle.load(open(output_path, 'rb'))
            prediction = predict(outputs, length)

            text = [ori[idx] for idx, bin in enumerate(prediction) if bin == 1]

            f.write('\nORIGINAL: ' + item['original'])
            f.write('\nCOMPRESS: ' + item['compressed'])
            f.write('\nPREDICT : ' + ' '.join(text))
    f.close()


def evaluate_all(ex_id, dataset):
    print('Load test data from %s' % (dataset['test']))
    test_data = load_conll_data_as_dict(dataset['test'])
    print('No.samples: %d' % (len(test_data)))
    output_path = os.path.join(endata.PROJECT_DIR, ex_id)
    epochs = [int(i) for i in os.listdir(output_path)]
    for folder_name in sorted(epochs):
        epoch_output_path = os.path.join(output_path, str(folder_name), 'output.pickle')
        evaluate(test_data, epoch_output_path)
        print('-')
        # except:
        #     print('Failed at epoch: %s' % (folder_name))
        # # evaluate(test_data, epoch_output_path)
        # print('-')


def evaluate_one(ex_id, epoch, dataset):
    test_data = load_conll_data_as_dict(dataset['test'])
    epoch_output_path = os.path.join(endata.PROJECT_DIR, ex_id, epoch, 'output.pickle')
    print('Output file: %s' % (epoch_output_path))
    evaluate(test_data, epoch_output_path)
    print('-')


eos = get_eos_vector()
rare = get_rare_vector()


def load_glove_vector():
    try:
        path = 'data/glove.6B.100d.pickle.bin'
        with open(path, 'rb') as f:
            word2vec  = pickle.load(f)
        return word2vec
    except:
        path = 'data/glove.6B.100d.txt'
        word2vec = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                items = line.replace('\r', '').replace('\n', '').split(' ')
                if len(items) < 10: continue
                word = items[0]
                vect = np.array([float(i) for i in items[1:] if len(i) > 1])
                word2vec[word] = vect
        path = 'data/glove.6B.100d.pickle.bin'
        with open(path, 'wb') as f:
            pickle.dump(word2vec, f)
        return word2vec
