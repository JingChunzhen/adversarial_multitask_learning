import tensorflow as tf
import yaml
from lstm_model import RNN
from discriminator import Discriminator

with open("../config/config.yaml", "rb") as f:
    params = yaml.load(f)


class Adversial_Network():
    def __init__(self):

        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")
        self.domain = tf.placeholder(
            tf.int32, [None, num_domains], name="domain")

        # trim the sequence to a fitted length according to specific task
        # how to update specific model -> tensorflow example GAN
        # use multi-thread to process shared and private model simultaneously
        # and test the function        

        private_model = []
        for task in params["task"]:
            private_model.append(RNN())
        shared_model = RNN()

        if embedding_init:
            with tf.name_scope("embedding-layer-with-glove-initialized"):
                self.W = tf.get_variable(shape=[vocab_size, embedding_size], initializer=tf.constant_initializer(
                    embedding_matrix), name='W', trainable=not static)
                # initializer can only be constant value or list with N-dimension
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
        else:
            with tf.name_scope("embedding-layer"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
        
        shared_model.rnn_output
        private_model[domain].rnn_output
