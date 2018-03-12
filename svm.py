import os, pickle, time, io
import argparse, traceback
from os.path import join as join
import sklearn.svm as svm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC
import numpy as np
import utils
from endata import CompressionData
from pyrouge import Rouge155


def load_feature(file_path, gold_path):
    gold_data = utils.load_conll_data_as_dict(gold_path)
    f = open(file_path, 'rb')
    outputs = pickle.load(f)
    x = []
    y = []
    for name, output in outputs.items():
        item = gold_data[name]
        bin = item['bin']
        y += bin
        for idx in range(len(bin)):
            x.append(output[idx].reshape((-1)))
    return x, y


def svm_tuning(path):
    global dataset
    print('Load train data')
    train_x, train_y = load_feature(os.path.join(path, 'train.feature'), dataset['train'])
    print('Load dev data:')
    dev_x, dev_y = load_feature(os.path.join(path, 'dev.feature'), dataset['dev'])
    print('Load test data:')
    test_x, test_y = load_feature(os.path.join(path, 'test.feature'), dataset['test'])

    start = time.time()
    c_max = 0
    f1_max = 0
    c = 1e-8
    while (c < 1e-3):
        print('----------------' + str(c) + '------------------')
        start = time.time()
        clf = svm.LinearSVC(C=c, class_weight='balanced')
        clf.fit(train_x, train_y)

        print("Train time: " + str(time.time() - start))
        start = time.time()

        validation = clf.predict(dev_x)
        prediction = clf.predict(test_x)

        print("Valid time: " + str(time.time() - start))
        pre_recall_fscore = precision_recall_fscore_support(dev_y, validation)
        acc = accuracy_score(dev_y, validation)
        print(pre_recall_fscore[2], acc)

        print("Testing time: " + str(time.time() - start))
        pre_recall_fscore = precision_recall_fscore_support(test_y, prediction)
        acc = accuracy_score(test_y, prediction)
        print(pre_recall_fscore[2], acc)

        if pre_recall_fscore[2][1] > f1_max:
            f1_max = pre_recall_fscore[2][1]
            c_max = c
            model = clf

        clf = 1
        c *= 2

    print(f1_max)
    print(c_max)
    return model


def select_from_model(path, c=2e-6):
    global dataset
    print('Load train data')
    train_x, train_y = load_feature(os.path.join(path, 'train.feature'), dataset['train'])
    print('Load dev data:')
    dev_x, dev_y = load_feature(os.path.join(path, 'dev.feature'), dataset['dev'])

    print('Load test data:')
    test_x, test_y = load_feature(os.path.join(path, 'test.feature'), dataset['test'])

    lsvc = LinearSVC(C=c, penalty="l1", dual=False).fit(train_x, train_y)
    model = SelectFromModel(lsvc, prefit=True)
    train_x_new = model.transform(train_x)
    test_x_new = model.transform(test_x)

    lsvc2 = LinearSVC(C=c, penalty="l1", dual=False).fit(train_x_new, train_y)
    prediction = lsvc2.predict(test_x_new)

    pre_recall_fscore = precision_recall_fscore_support(test_y, prediction)
    acc = accuracy_score(test_y, prediction)
    print(pre_recall_fscore[2], acc)


def evaluate(epoch_output_path, model):
    global dataset
    test_data = utils.load_conll_data_as_dict(dataset['test'])

    def save(words, bin, file_path):
        with io.open(file_path, 'w', encoding='utf8') as f:
            for idx, label in enumerate(bin):
                if label == 1:
                    f.write(words[idx] + ' ')
            if bin[-1] == 0 and words[-1] == '.':
                f.write(u'.')

    output_file = os.path.join(epoch_output_path, 'output.pickle')
    with open(output_file, 'rb') as f:
        outputs = pickle.load(f)
    output_folder = epoch_output_path.replace('/', '-')
    utils.create_folder(output_folder)
    reference_folder = os.path.join(output_folder, 'reference')
    system_folder = os.path.join(output_folder, 'system')
    utils.create_folder(reference_folder)
    utils.create_folder(system_folder)

    keep_predict = 0
    total_word = 0
    for name, _features in outputs.items():
        target_labels = test_data[name]['bin']
        sentence_length = len(target_labels)
        sentence_features = []
        for idx in range(sentence_length):
            sentence_features.append(_features[idx].reshape((-1)))

        prediction = model.predict(sentence_features)
        keep_predict += sum(prediction)
        total_word += sentence_length
        ori_words = test_data[name]['word']
        if ori_words[-1] == '.':
            prediction[-1] = 1

        f = os.path.join(reference_folder, '%s_reference.txt' % (name))
        save(ori_words, target_labels, f)

        f = os.path.join(system_folder, '%s_system.txt' % (name))
        save(ori_words, prediction, f)
    eval_by_rouge(output_folder)

    keep_predict += 1e-10
    print('Compression rate: %f' % (keep_predict / total_word))


def eval_by_rouge(output_path):
    r = Rouge155()
    r.system_dir = join(output_path, 'system')
    r.model_dir = join(output_path, 'reference')
    r.system_filename_pattern = '(\d+)_system.txt'
    r.model_filename_pattern = '#ID#_reference.txt'

    output = r.convert_and_evaluate()
    print('----------------------------------------------')
    print('Output %s' % (output_path))
    print(output)


parser = argparse.ArgumentParser()
parser.add_argument('-tune', help='Tuning SVM classifier',
                    action='store_true', default=False)
parser.add_argument('-eval', help='Select feature',
                    action='store_true', default=False)

parser.add_argument('-i', '--input', help='Epoch output folder', default='e118_4/4')

arg = parser.parse_args()
dataset = CompressionData.mikolov_conll
print('Dataset: %s' % (dataset['name']))

if arg.tune:
    svm_tuning(arg.input)
elif arg.eval:
    model = svm_tuning(arg.input)
    evaluate(arg.input, model)
