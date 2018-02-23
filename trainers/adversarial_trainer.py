import os
import sys
sys.path.append("..")
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from model.adversial_model import Adversarial_Model
from utils.data_loader import load_data, batch_iter

with open("", "rb") as f:
    params = yaml.load(f)


class EVAL(object):
    """
    to be continued 
    """
    def __init__(self, sequence_length):
        # load data first
        self.processor = learn.preprocessing.VocabularyProcessor.restore(
            "../temp/vocab")
        self.processor.max_document_length = sequence_length
        raw_data, self.train_label = load_data("train")
        self.train_data = []
        for d in raw_data:
            # for each task in data
            d = list(self.processor.transform(d))  # generator -> list
            self.train_data.append(d)
        del raw_data
        self.test_data = []
        raw_data, self.test_label = load_data("test")
        for d in raw_data:
            d = self.processor.transform(d)
            self.test_data.append(d)
        del raw_data
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

    def process(self):
        """
        """
        embedding_matrix = list(chain.from_iterable(
            self.embedding_matrix)) if self.embedding_matrix else None
        with tf.Graph().as_default():
            instance = Adversarial_Model(sequence_length=params["global"]["sequence_length"],
                                         num_classes=params["global"]["num_classes"],
                                         embedding_size=params["global"]["embedding_size"],
                                         vocab_size=len(
                                             self.processor.vocabulary_),
                                         embedding_matrix=embedding_matrix,
                                         static=params["global"]["static"],
                                         rnn_hidden_size=params["global"]["rnn_hidden_size"],
                                         shared_num_layers=params["shared_model"]["num_layer"],
                                         private_num_layers=params["private_model"]["num_layer"]
                                         dynamic=params["global"]["dynamic"],
                                         use_attention=params["global"]["use_attention"],
                                         attention_size=params["global"]["attention_size"],
                                         mlp_hidden_size=params["global"]["hidden_size"])
            global_step = tf.Variable(0, trainable=False)
            init = tf.global_variables_initializer()

            adv_loss = instance.adv_loss # TODO
            diff_loss = instance.diff_loss
            task_loss = instance.task_loss

            discriminator_accuracy = instance.discriminator_accuracy
            task_accuracy = instance.task_accuracy

            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate)
            task_optimizer = tf.train.AdamOptimizer(learning_rate)

            discriminator_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            task_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}-rnn".format(params["task"][task]))
            shared_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared")

            discriminator_train_op = discriminator_optimizer.minimize(
                adv_loss, var_list=discriminator_vars)
            task_train_op = task_optimizer.minimize(
                task_loss+diff_loss, var_list=task_vars.extend(shared_vars)) # TODO
 
            with tf.Session() as sess:
                def train_step(x_batch, y_batch):
                    feed_dict = {
                        instance.input_x: x_batch,
                        instance.input_y: y_batch
                    }
                    sess.run(
                        [adv_loss, diff_loss, task_loss]
                    )

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        instance.input_x: x_batch,
                        instance.input_y: y_batch
                    }
                sess.run(init)

                for task, batch in batch_iter(list(zip(self.x_train, self.y_train)), batch_size, epochs):
                    x_batch, y_batch = zip(*batch)
                    instance.process(task)
                    pass
