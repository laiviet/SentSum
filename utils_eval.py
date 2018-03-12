import os
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def load_data(file_path):
    f = open(file_path, 'rb')
    word = pickle.load(f)
    stem = pickle.load(f)
    bin = pickle.load(f)
    f.close()
    return word, stem, bin


def predict(output, length):
    prediction = []
    # input('Continute: ')
    for i in range(length):
        if output[i][0][0] > output[i][0][1]:
            prediction.append(0)
        else:
            prediction.append(1)
            #   print(output[i][0][0], output[i][0][1])
    return prediction


def evaluate(output_path, printresult=True):
    source_data = 'data/parsedata/'
    files = os.listdir(output_path)

    total_target = []
    total_prediction = []
    for fname in files:
        names = fname.split('.')
        if (len(names) != 3):
            continue
        if names[2] == 'data':
            foutput = open(output_path + '/' + fname, 'rb')
            output = pickle.load(foutput)
            word, stem, bin = load_data(source_data + fname)

            l = len(word)
            prediction = predict(output, l)
            total_target = total_target + bin
            total_prediction = total_prediction + prediction

    pre_recall_fscore = precision_recall_fscore_support(total_target, total_prediction)
    acc = accuracy_score(total_target, total_prediction)
    if printresult:
        print(pre_recall_fscore[2], acc)
    return (pre_recall_fscore, acc)