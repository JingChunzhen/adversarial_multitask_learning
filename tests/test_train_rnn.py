import re
import yaml
import sys
sys.path.append("..")
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn

from model.rnn_model import RNN


with open("../config/config.yaml", "rb") as f:
    params = yaml.load(f)


class TEST_RNN(object):
    """
    this is a test to test whether the RNN model could work 
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 embedding_matrix,
                 static,
                 hidden_size,
                 num_layers,
                 dynamic,
                 use_attention,
                 attention_size
                 ):
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")

        with tf.name_scope("embedding-layer"):
            self.W = tf.get_variable(shape=[vocab_size, embedding_size],
                                     initializer=tf.constant_initializer(
                embedding_matrix),
                name='W',
                trainable=not static)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("calculate-sequence-length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("rnn-processing"):
            self.rnn_model = RNN(sequence_length,
                                 hidden_size,
                                 num_layers,
                                 dynamic=True,
                                 use_attention=True,
                                 attention_size=attention_size)
            output = self.rnn_model.process(
                self.embedded_chars, seq_len, "rnn-model")

        with tf.name_scope("fully-connected-layer"):
            w = tf.Variable(tf.truncated_normal(
                [hidden_size*2, num_classes], stddev=0.1), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(output, w, b)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y, logits=scores)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            predictions = tf.argmax(scores, 1, name="predictions")
            correct_predictions = tf.equal(
                predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    this is very crude
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(task, train_or_test):
    """
    Arg:
        task (string): specific task
        train_or_test (string): can either be train or test
    Returns:
        data (list), label (list)
    """
    data = []
    label = []
    for task in params["task"]:
        file_in = "../data/mtl-dataset/{}.task.{}".format(task, train_or_test)
        reviews = []
        polarities = []
        with open(file_in, "r", encoding='ISO-8859-1') as f:
            for line in f.readlines():
                line = clean_str(line)
                polarities.append([1, 0] if int(line[0]) == 0 else [0, 1])
                review = line[1:].strip()
                reviews.append(review)
        data.append(reviews)
        label.append(polarities)
    return data, label


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    need to be tested 
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class EVAL(object):
    def __init__(self, sequence_length):
        self.processor = learn.preprocessing.VocabularyProcessor.restore(
            "../temp/vocab")
        self.processor.max_document_length = sequence_length
        raw_data, raw_label = load_data("apperal", "train")
        self.train_data = []
        self.train_label = []
        for rd, rl in zip(raw_data, raw_label):
            # for each task in data
            tmp_data = []
            tmp_label = []
            rd = list(self.processor.transform(rd))  # generator -> list
            for tmp_x, tmp_y in zip(rd, rl):
                tmp_x = tmp_x.tolist()
                if np.sum(tmp_x) != 0:
                    tmp_data.append(tmp_x)
                    tmp_label.append(tmp_y)
            self.train_data.append(tmp_data)
            self.train_label.append(tmp_label)
        del raw_data, raw_label
        print("load training data complete!")
        self.embedding_matrix = self._embedding_matrix_initializer()

    def _embedding_matrix_initializer(self):
        """
        embedding layer initialization using pre-trained glovec
        """
        file_wv = "../data/glove.6B/glove.6B.{}d.txt".format(
            params["global"]["embedding_size"])
        wv = {}
        embedding_matrix = []

        with open(file_wv, 'r') as f:
            for line in f:
                line = line.split(' ')
                word = line[0]
                wv[word] = list(map(float, line[1:]))

        for idx in range(len(self.processor.vocabulary_)):
            word = self.processor.vocabulary_.reverse(idx)
            embedding_matrix.append(
                wv[word] if word in wv else np.random.normal(size=params["global"]["embedding_size"]))
        return embedding_matrix

    def process(self, learning_rate, batch_size, epochs, evaluate_every):
        embedding_matrix = list(chain.from_iterable(
            self.embedding_matrix)) if self.embedding_matrix else None
        with tf.Graph().as_default():
            test_rnn = TEST_RNN(
                sequence_length=params["global"]["sequence_length"],
                num_classes=params["global"]["num_classes"],
                embedding_size=params["global"]["embedding_size"],
                vocab_size=len(
                    self.processor.vocabulary_),
                embedding_matrix=embedding_matrix,
                static=params["global"]["static"],
                hidden_size=params["global"]["rnn_hidden_size"],
                num_layers=params["shared_model"]["num_layers"],
                dynamic=params["global"]["dynamic"],
                use_attention=params["global"]["use_attention"],
                attention_size=params["global"]["attention_size"])

            global_step = tf.Variable(0, trainable=False)
            init = tf.global_variables_initializer()

            loss = test_rnn.loss
            acc = test_rnn.accuracy

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)

            with tf.Session() as sess:
                sess.run(init)

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        test_rnn.input_x: x_batch,
                        test_rnn.input_y: y_batch
                    }
                    step, _, loss_, acc_ = sess.run(
                        [
                            global_step,
                            train_op,
                            loss,
                            acc
                        ], feed_dict=feed_dict
                    )
                    return step, loss_, acc_

                def dev_step(x_batch, y_batch):
                    feed_dict = {
                        instance.input_x: x_batch,
                        instance.input_y: y_batch
                    }
                    step, loss_, acc_ = sess.run(
                        [
                            global_step,
                            loss,
                            acc
                        ], feed_dict=feed_dict
                    )
                    return step, loss_, acc_

                for batch in batch_iter(list(zip(self.train_data, self.train_label)), batch_size, epochs):
                    x_batch, y_batch = zip(*batch)
                    current_step, loss_, acc_ = train_step(
                        x_batch, y_batch)

                    print("step: {}, task loss: {:.5f}, task acc: {:.2f}".format(
                        current_step, loss_, acc_))
                    if current_step % evaluate_every == 0:
                        """
                        test transfer effect
                        """
                        pass
                    pass
            pass


if __name__ == "__main__":
    # TODO ValueError: setting an array element with a sequence.
    eval = EVAL(params["global"]["sequence_length"])
    eval.process(
        learning_rate=params["global"]["learning_rate"],
        batch_size=params["global"]["batch_size"],
        epochs=params["global"]["epochs"],
        evaluate_every=100
    )
