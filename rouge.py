import os, sys, argparse, io, traceback
import pickle
from endata import CompressionData
import utils
from pyrouge import Rouge155
import baseline_utils as blutils



def predict(output, length):
    prediction = []
    # input('Continute: ')
    for i in range(length):
        if output[i][0][0] > output[i][0][1]:
            prediction.append(0)
        else:
            prediction.append(1)
    return prediction


def to_file_baseline(ex_output_path):
    def save(words, bin, file_path):
        with io.open(file_path, 'w', encoding='utf8') as f:
            for idx, label in enumerate(bin):
                if label == 1:
                    f.write(words[idx] + ' ')
            if bin[-1] == 0 and words[-1] == '.':
                f.write(u'.')

    for epoch_name in os.listdir(ex_output_path):
        try:
            epoch_output_path = os.path.join(ex_output_path, epoch_name)
            output_file = os.path.join(epoch_output_path, 'output.pickle')
            with open(output_file, 'rb') as f:
                outputs = pickle.load(f)
            output_folder = epoch_output_path.replace('/', '-')
            utils.create_folder(output_folder)
            reference_folder = os.path.join(output_folder, 'reference')
            system_folder = os.path.join(output_folder, 'system')
            utils.create_folder(reference_folder)
            utils.create_folder(system_folder)

            for name, labels in outputs.items():
                target_labels = test_data[name]['bin']
                prediction = predict(labels, len(target_labels))
                ori_words = test_data[name]['word']
                if ori_words[-1] == '.':
                    prediction[-1] = 1

                f = os.path.join(reference_folder, '%s_reference.txt' % (name))
                save(ori_words, target_labels, f)

                f = os.path.join(system_folder, '%s_system.txt' % (name))
                save(ori_words, prediction, f)
            eval_by_rouge(output_folder)
        except:
            print('[Error] Epoch: %s'%(epoch_name))
            traceback.print_exc()




def to_file(ex_output_path, test_data):
    def save(words, bin, file_path):
        with io.open(file_path, 'w', encoding='utf8') as f:
            for idx, label in enumerate(bin):
                if label == 1:
                    f.write(words[idx] + ' ')
            if bin[-1] == 0 and words[-1] == '.':
                f.write(u'.')

    for epoch_name in os.listdir(ex_output_path):
        try:
            epoch_output_path = os.path.join(ex_output_path, epoch_name)
            output_file = os.path.join(epoch_output_path, 'output.pickle')
            with open(output_file, 'rb') as f:
                outputs = pickle.load(f)
            output_folder = epoch_output_path.replace('/', '-')
            utils.create_folder(output_folder)
            reference_folder = os.path.join(output_folder, 'reference')
            system_folder = os.path.join(output_folder, 'system')
            utils.create_folder(reference_folder)
            utils.create_folder(system_folder)

            for name, labels in outputs.items():
                target_labels = test_data[name]['bin']
                prediction = predict(labels, len(target_labels))
                ori_words = test_data[name]['word']
                if ori_words[-1] == '.':
                    prediction[-1] = 1

                f = os.path.join(reference_folder, '%s_reference.txt' % (name))
                save(ori_words, target_labels, f)

                f = os.path.join(system_folder, '%s_system.txt' % (name))
                save(ori_words, prediction, f)
            eval_by_rouge(output_folder)
        except:
            print('[Error] Epoch: %s'%(epoch_name))
            traceback.print_exc()

def eval_by_rouge(output_path):
    r = Rouge155()
    r.system_dir = os.path.join(output_path, 'system')
    r.model_dir = os.path.join(output_path, 'reference')
    r.system_filename_pattern = '(\d+)_system.txt'
    r.model_filename_pattern = '#ID#_reference.txt'

    output = r.convert_and_evaluate()
    print('----------------------------------------------')
    print('Output %s'%(output_path))
    print(output)


parser = argparse.ArgumentParser()
parser.add_argument('-all', help='Automatic restore from saved epoch.',
                    action='store_true', default=False)
parser.add_argument('-eval', help='Automatic restore from saved epoch.',
                    action='store_true', default=False)

parser.add_argument('-i', '--input', help='System output')
arg = parser.parse_args()

if arg.all:
    dataset = CompressionData.fillipova_conll
    print('Dataset: %s' % (dataset['name']))
    test_data = utils.load_conll_data_as_dict(dataset['test'])
    to_file(arg.input, test_data)
elif arg.eval:
    eval_by_rouge(arg.input)
