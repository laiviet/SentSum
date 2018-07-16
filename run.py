from __future__ import print_function
# -*- coding: utf-8 -*-
#
#
#   Author: Lai Dac Viet
#   Model: Bidirectional Sequence-to-Sequence LSTM
#   Tensorflow: 0.12.0
#
import os, sys, time, pickle, argparse
import numpy as np
import tensorflow as tf
import utils
import endata
from endata import CompressionData
from model import BiEncoderDecoderModel


flags = tf.flags
flags.DEFINE_integer("max_len", 192, "Maximum length of sentence")
# flags.DEFINE_integer("max_len", 12, "Maximum length of sentence")
flags.DEFINE_integer("embedding_size", 100, "Dimension of embedding vectos")
flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.8, 'Initial learning rate.')
flags.DEFINE_integer('class_size', 2, 'Number of steps to run trainer.')
FLAGS = flags.FLAGS


def get_sample(model, data, index, FLAGS=FLAGS):
    sample = data[index]
    word = sample['word']
    stem = sample['stem']
    bin = sample['bin']
    name = sample['name']
    eos = utils.get_eos_vector()

    fw = np.ndarray(shape=(0, 1, FLAGS.embedding_size), dtype=float)
    labels = np.ndarray(shape=(0, 1, FLAGS.class_size), dtype=float)
    eos_ex = np.ndarray(shape=(0, 1, FLAGS.embedding_size), dtype=float)

    l = len(word)
    for i in range(l):
        _word = word[i]  # [2:len(word[i]) - 1]
        _stem = stem[i]  # [2:len(stem[i]) - 1]
        vector = utils.get_word_vector(model, _word, _stem)
        fw = np.append(fw, vector, axis=0)

    for i in range(l, FLAGS.max_len):
        eos_ex = np.append(eos_ex, eos, axis=0)
        bin.append(0)

    fw = np.append(fw, eos_ex, axis=0)

    for i in range(FLAGS.max_len):
        if bin[i] == 0:
            x = np.array([[[1, 0]]])
        elif bin[i] == 1:
            x = np.array([[[0, 1]]])
        labels = np.append(labels, x, axis=0)

        # print(d.shape, e.shape, labels.shape)
    return fw, labels, name, np.array([l])


def save_output(path, predicts):
    f = open(path, 'wb')
    pickle.dump(predicts, f)
    f.close()


def train(ex_id, restore=False):
    global dataset
    output_path = ex_id + '/'
    print(output_path)
    utils.create_folder(output_path)
    train_data = utils.load_conll_data(dataset['train'])
    train_range = len(train_data)
    print('Train dataset: %d' % (train_range))

    word2vec_model = utils.load_word2vec()
    global FLAGS, tf_config
    with tf.Graph().as_default(), tf.Session() as session:
        # with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("RNN", reuse=None, initializer=initializer):
            utils.log("Building model.. ", ex_id)
            i_train = BiEncoderDecoderModel(is_training=True, FLAGS=FLAGS)
            start_epoch = 0
            if restore:
                start_epoch = max(int(i) for i in os.listdir(arg.id)) - 1
                if start_epoch > 0:
                    print('Restoring model: %s...' % (start_epoch))
                    model_file_path = os.path.join(ex_id, str(start_epoch), 'model.ckpt')
                    i_train.saver.restore(session, model_file_path)
                else:
                    utils.log('No saved model, initialize all variables...', ex_id)
                    tf.global_variables_initializer().run()
            else:
                tf.global_variables_initializer().run()
            for epoch in range(start_epoch + 1, 150):
                print("Epoch: %d" % (epoch), ex_id)
                epoch_output_path = os.path.join(output_path, str(epoch))
                utils.create_folder(epoch_output_path)

                train_cost = 0.0
                per = np.random.permutation(train_range)
                for i, index in enumerate(per):
                    inputs, labels, name, sentence_len = get_sample(word2vec_model, train_data, index)
                    start = time.time()
                    cost, predicts, feature = i_train.train(session, inputs, labels, sentence_len)
                    train_cost += cost
                    if i % 100 == 0:
                        print('Time: %f\r' % ((time.time() - start) / 100), end='')
                        utils.update_epoch(epoch, i, ex_id)
                print('Train: ' + str(train_cost), ex_id)
                model_file_path = os.path.join(epoch_output_path, 'model.ckpt')
                i_train.save_model(session, model_file_path)


def test(ex_id, start_epoch, valid=False):
    global dataset, word_embedding
    if valid:
        test_data = utils.load_conll_data_as_dict(dataset['dev'])
        test_range = len(test_data)
        print("Valid set: %d" % (test_range))
    else:
        test_data = utils.load_conll_data_as_dict(dataset['test'])
        test_range = len(test_data)
        print("Test set: %d" % (test_range))

    global FLAGS, tf_config
    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("RNN", reuse=False, initializer=initializer):
            print('Building model..')
            i_test = BiEncoderDecoderModel(is_training=False, FLAGS=FLAGS)
            for epoch in range(start_epoch, 150):
                epoch_output_path = os.path.join(ex_id, str(epoch))
                print('Restoring model: %d...' % (epoch))
                model_file_path = os.path.join(epoch_output_path, 'model.ckpt')
                i_test.saver.restore(session, model_file_path)

                # Test model
                test_cost = 0.0
                print('Starting test..')
                start = time.time()
                output = {}
                for name, sample in test_data.items():
                    inputs, labels, _, sentence_len = get_sample(word_embedding, [sample], 0)
                    cost, predicts, feature = i_test.test(session, inputs, labels, sentence_len)
                    test_cost += cost
                    output[name] = predicts
                if valid:
                    file_path = os.path.join(epoch_output_path, 'dev.pickle')
                else:
                    file_path = os.path.join(epoch_output_path, 'output.pickle')
                save_output(file_path, output)
                print('Finished: ' + str(time.time() - start))
                utils.log('Loss of epoch [%d]: %s' % (epoch, str(test_cost)), ex_id)


def make_feature(ex_id, epoch):
    global dataset
    train_data = utils.load_conll_data_as_dict(dataset['train'])
    dev_data = utils.load_conll_data_as_dict(dataset['dev'])
    test_data = utils.load_conll_data_as_dict(dataset['test'])
    word2vec_model = utils.load_word2vec()

    global FLAGS, tf_config
    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("RNN", reuse=False, initializer=initializer):
            print('Building model..')
            i_test = BiEncoderDecoderModel(is_training=False)

            epoch_output_path = os.path.join(ex_id, epoch)
            print('Restoring model: %s...' % (epoch))
            model_file_path = os.path.join(epoch_output_path, 'model.ckpt')
            i_test.saver.restore(session, model_file_path)

            # Test model
            test_cost = 0.0
            print('Starting test..')
            start = time.time()

            for idx, data in enumerate([train_data, dev_data, test_data]):
                output = {}
                for name, sample in data.items():
                    inputs, labels, _, sentence_len = get_sample(word_embedding, [sample], 0)
                    cost, predicts, feature = i_test.test(session, inputs, labels, sentence_len)
                    test_cost += cost
                    output[name] = predicts
                if idx == 0:
                    print('Saving train feature')
                    file_path = os.path.join(epoch_output_path, 'train.feature')
                elif idx == 1:
                    print('Saving dev feature')
                    file_path = os.path.join(epoch_output_path, 'dev.feature')
                else:
                    print('Saving test feature')
                    file_path = os.path.join(epoch_output_path, 'test.feature')
                save_output(file_path, output)

    print('DONE!')


def load_data(arg):
    if arg.glove:
        utils.log('Loading glove..', arg.id)
        word_embedding = utils.load_glove_vector()
    else:
        utils.log('Loading word2vec..', arg.id)
        word_embedding = utils.load_word2vec()
    return word_embedding


parser = argparse.ArgumentParser()
parser.add_argument('-train', help='Train with the given ID',
                    action='store_true', default=False)
parser.add_argument('-valid', help='Valid the E-th epoch of the I experiment',
                    action='store_true', default=False)
parser.add_argument('-restore', help='Automatic restore from saved epoch.',
                    action='store_true', default=False)
parser.add_argument('-test', help='Test the E-th epoch of the I experiment',
                    action='store_true', default=False)
parser.add_argument('-eval', help='Evaluate the E-th epoch of the I experiment.',
                    action='store_true', default=False)
# parser.add_argument('-save', help='Print the output of E-th epoch of the I experiment.',
#                    action='store_true', default=False)
parser.add_argument('-feature', help='Create feature of E-th epoch of the I experiment',
                    action='store_true', default=False)
parser.add_argument('-debug', help='Using debug dataset', action='store_true', default=False)
parser.add_argument('-glove', help='Using GloVe word representation', action='store_true', default=False)

parser.add_argument('-i', '--id', help='Output directory', default='output')
parser.add_argument('-e', '--epoch', help='Experiment epoch', default=-1)
arg = parser.parse_args()

if arg.debug:
    print('Use debug data')
    dataset = CompressionData.d_fillipova_conll
else:
    dataset = CompressionData.fillipova_conll

if arg.train:
    word_embedding = load_data(arg)
    train(arg.id, arg.restore)
elif arg.valid:
    word_embedding = load_data(arg)
    if arg.epoch == -1:
        test(arg.id, 1, valid=True)
    else:
        test(arg.id, int(arg.epoch), valid=True)
elif arg.test:
    word_embedding = load_data(arg)
    if arg.epoch == -1:
        test(arg.id, 1)
    else:
        test(arg.id, int(arg.epoch))
# elif arg.save:
#     epoch_output_path = os.path.join(endata.PROJECT_DIR, arg.id, arg.epoch)
#     testdata = utils.load_conll_data_as_dict(dataset['test'])
#     utils.print_result(testdata, epoch_output_path, dataset)
elif arg.eval:
    if arg.epoch == -1:
        utils.evaluate_all(arg.id, dataset)
    else:
        utils.evaluate_one(arg.id, arg.epoch, dataset)
elif arg.feature:
    word_embedding = load_data(arg)
    if arg.epoch == -1:
        print('\n[Warning] Assuming create feature for epoch: 1 to 10 ')
        for i in range(10):
            make_feature(arg.id, i)
    else:
        make_feature(arg.id, arg.epoch)
