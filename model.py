# -*- coding: utf-8 -*-
#
#
#   Author: Lai Dac Viet
#   Model: Bidirectional Sequence-to-Sequence LSTM
#   Tensorflow: 0.12.0
#
import os, sys, time, pickle, argparse
from tensorflow.python.ops import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops
import tensorflow as tf


def reverse_seq(input_seq, lengths):
    if lengths is None:
        return list(reversed(input_seq))

    input_shape = tensor_shape.matrix(None, None)
    for input_ in input_seq:
        input_shape.merge_with(input_.get_shape())
        input_.set_shape(input_shape)

    # Join into (time, batch_size, depth)
    s_joined = array_ops.pack(input_seq)

    # TODO(schuster, ebrevdo): Remove cast when reverse_sequence takes int32
    if lengths is not None:
        lengths = math_ops.to_int64(lengths)

    # Reverse along dimension 0
    s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops.unpack(s_reversed)
    for r in result:
        r.set_shape(input_shape)
    return result


# Gathering summary variables
def variable_summaries(var, name):
    ex_id = 'e118'
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope(ex_id + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(ex_id + '_mean/' + name, mean)
        with tf.name_scope(ex_id + '_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(ex_id + '_stddev/' + name, stddev)
        tf.summary.scalar(ex_id + '_max/' + name, tf.reduce_max(var))
        tf.summary.scalar(ex_id + '_min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


class BiEncoderDecoderModel(object):
    def __init__(self, is_training, FLAGS):
        self.sentence_len = tf.placeholder(tf.int64, shape=[1])
        self.embedding_inputs = tf.placeholder(tf.float32, shape=[FLAGS.max_len, 1, FLAGS.embedding_size])
        self.target_inputs = tf.placeholder(tf.float32, shape=[FLAGS.max_len, 1, FLAGS.class_size])

        inputs = []
        # targets = []

        for i in range(0, FLAGS.max_len):
            inputs.append(self.embedding_inputs[i, :, :])
        # for i in range(0, FLAGS.max_len):
        #     targets.append(self.target_inputs[i, :, :])

        targets = tf.reshape(self.target_inputs, (FLAGS.max_len, FLAGS.class_size))
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope("fw", reuse=None, initializer=initializer):
            lstm_fw1 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            lstm_fw2 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            lstm_fw3 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)

            drop_lstm_fw1 = tf.nn.rnn_cell.DropoutWrapper(lstm_fw1, output_keep_prob=FLAGS.keep_prob)
            drop_lstm_fw2 = tf.nn.rnn_cell.DropoutWrapper(lstm_fw1, output_keep_prob=FLAGS.keep_prob)
            if is_training:
                stacked_lstm_fw = tf.nn.rnn_cell.MultiRNNCell([drop_lstm_fw1, drop_lstm_fw2, lstm_fw3],
                                                              state_is_tuple=True)
            else:
                stacked_lstm_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw1, lstm_fw2, lstm_fw3],
                                                              state_is_tuple=True)

            reverse_input = reverse_seq(inputs, self.sentence_len)
            _, init_fw = rnn.rnn(stacked_lstm_fw, reverse_input, sequence_length=self.sentence_len,
                                 dtype=tf.float32)

        with tf.variable_scope("bw", reuse=None, initializer=initializer):
            lstm_bw1 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            lstm_bw2 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            lstm_bw3 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)

            drop_lstm_bw1 = tf.nn.rnn_cell.DropoutWrapper(lstm_bw1, output_keep_prob=FLAGS.keep_prob)
            drop_lstm_bw2 = tf.nn.rnn_cell.DropoutWrapper(lstm_bw1, output_keep_prob=FLAGS.keep_prob)

            if is_training:
                stacked_lstm_bw = tf.nn.rnn_cell.MultiRNNCell([drop_lstm_bw1, drop_lstm_bw2, lstm_bw3],
                                                              state_is_tuple=True)
            else:
                stacked_lstm_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_bw1, lstm_bw2, lstm_bw3],
                                                              state_is_tuple=True)
            _, init_bw = rnn.rnn(stacked_lstm_bw, inputs, sequence_length=self.sentence_len, dtype=tf.float32)

        bid_output, _, _ = tf.nn.bidirectional_rnn(stacked_lstm_fw, stacked_lstm_bw, inputs,
                                                   initial_state_fw=init_fw, initial_state_bw=init_bw,
                                                   dtype=tf.float32, sequence_length=self.sentence_len)

        self.outputs = tf.reshape(bid_output, (FLAGS.max_len, FLAGS.embedding_size * 2, 1))
        weight = [tf.get_variable("softmax_w", [FLAGS.embedding_size * 2, FLAGS.class_size],
                                  dtype=tf.float32)] * FLAGS.max_len

        self.logits = tf.batch_matmul(self.outputs, weight, adj_x=True, adj_y=False)
        logits = tf.reshape(self.logits, (-1, FLAGS.class_size))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='cross-entropy-loss')
        self.cost = tf.reduce_sum(loss) / tf.cast(self.sentence_len, tf.float32)

        self.saver = tf.train.Saver(max_to_keep=100)

        if is_training:
            variable_summaries(loss, "cross_entropy")
            self.merged = tf.summary.merge_all()
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            self.train_op = optimizer.minimize(self.cost)

    def train(self, session, inputs, labels, sentence_len):
        feed_dict = {
            self.embedding_inputs: inputs,
            self.target_inputs: labels,
            self.sentence_len: sentence_len
        }
        predicts, cost, _, feature = session.run([self.logits, self.cost, self.train_op, self.outputs], feed_dict)
        return cost, predicts, feature

    def valid(self, session, inputs, labels, sentence_len):
        feed_dict = {
            self.embedding_inputs: inputs,
            self.target_inputs: labels,
            self.sentence_len: sentence_len
        }
        summary, predicts, cost, feature = session.run([self.merged, self.logits, self.cost, self.outputs], feed_dict)
        return summary, cost, predicts, feature

    def test(self, session, inputs, labels, sentence_len):
        feed_dict = {
            self.embedding_inputs: inputs,
            self.target_inputs: labels,
            self.sentence_len: sentence_len
        }
        predicts, cost, feature = session.run([self.logits, self.cost, self.outputs], feed_dict)
        return cost, predicts, feature

    def save_model(self, session, path):
        _ = self.saver.save(session, path)
