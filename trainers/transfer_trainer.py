import os
import sys
sys.path.append("..")
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import classification_report
from model.adversarial_model import Adversarial_Network
from model.transfer_model import Transfer
from utils.data_loader import load_data, batch_iter
from itertools import chain

with open("../config/config.yaml", "rb") as f:
    params = yaml.load(f)


class EVAL(object):
    """
    transfer learning using shared model in adversarial network
    """

    def __init__(self):
        """
        using embedding_matrix, shared-model and fully-conneted params to initialize the transfer model 
        Args:
            embedding_matrix (list): embedding matrix of pre-trained adversarial model  
            fc_w (list): fully connected weight of pre-trained model
            fc_b (list): fully connected bias of pre-trained model         
        """
        pass

        
    def _params_initializer(self, model_path):
        """
        initialize parameters of the transfer model using pre-trained adversarial model 
        this is supposed to be used in trainers 
        Args:
            model_path (string): path stored pre-trained adversarial model 
        Returns:
            embedding_matrix (list)
            fc_params: fully-connected weights and bias 
        """
        adn = Adversarial_Network(
            sequence_length=params["global"]["sequence_length"],
            num_classes=params["global"]["num_classes"],
            embedding_size=params["global"]["embedding_size"],
            vocab_size=len(
                self.processor.vocabulary_),
            embedding_matrix=embedding_matrix,
            static=params["global"]["static"],
            rnn_hidden_size=params["global"]["rnn_hidden_size"],
            shared_num_layers=params["shared_model"]["num_layers"],
            private_num_layers=params["private_model"]["num_layers"],
            dynamic=params["global"]["dynamic"],
            use_attention=params["global"]["use_attention"],
            attention_size=params["global"]["attention_size"],
            mlp_hidden_size=params["global"]["mlp_hidden_size"])
        saver = tf.train.Saver()

        embedding_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="W")
        shared_model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")
        fc_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="fully-connected-layer")

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_path)
            embedding_matrix = sess.run(embedding_vars)
            fc_params = sess.run(fc_vars)
            shared_model_vars = sess.run(shared_model_vars)

        embedding_matrix = list(chain.from_iterable(embedding_matrix))
        fc_w, fc_b = fc_params
        fc_w = fc_w[:rnn_hidden_size * 2]
        fc_w = list(chain.from_iterable(fc_w))
        fc_b = list(chain.from_iterable(fc_b))
        return embedding_matrix, fc_w, fc_b, shared_model_vars

    def _rnn_initializer(self, shared_model_vars):
        """
        TODO: need to be tested 
        initialize the rnn model in transfer model using pre-trained adversarial model
        Args:            
            shared_model_vars (list): shared model's variables 
        """
        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="transfer-shared")
        for rnn_var, shared_model_var in zip(rnn_vars, shared_model_vars):
            tf.assign(rnn_var, shared_model_var)                                

    def process(self, learning_rate):
        """
        incremental learning based on the pre-trained adversarial model 
        """
        embedding_matrix, fc_w, fc_b, shared_model_vars = self._params_initializer(model_path)
        instance = Transfer(
            sequence_length=params["global"]["sequence_length"],
            num_classes=params["global"]["num_classes"],
            embedding_size=params["global"]["embedding_size"],
            vacab_size=len(embedding_matrix) / params["global"]["embedding_size"],
            static=params["global"]["static"],
            rnn_hidden_size=params["global"]["rnn_hidden_size"],
            num_layers=params["shared"]["num_layers"],
            dynamic=params["global"]["dynamic"],
            use_attention=params["global"]["use_attention"],
            attention_size=params["global"]["attention_size"],
            embedding_matrix=embedding_matrix,
            fc_w=fc_w,
            fc_b=fc_b)  
        
        #instance.rnn_model.cell_bw.weights
        #instance.rnn_model.cell_fw.weights
        global_step = tf.Variable(0, trainable=False)
        loss = instance.task_loss
        accuracy = instance.task_accuracy
        preds = instance.predictions

        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("loss", loss)

        merged_summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(self._rnn_initializer(shared_model_vars))

            train_summary_writer = tf.summary.FileWriter(
                logdir='../temp/summary/transfer/train', graph=sess.graph)
            dev_summary_writer = tf.summary.FileWriter(
                logdir='../temp/summary/transfer/dev', graph=sess.graph)
            def train_step(batch_x, batch_y):
                feed_dict = {
                    instance.input_x: batch_x,
                    instance.input_y: batch_y}
                step, summary, _, loss_, accuracy_ = sess.run(
                    [
                        global_step, 
                        merged_summary_op,
                        train_op,
                        loss,
                        accuracy
                    ], feed_dict=feed_dict)
                
                train_summary_writer.add_summary(summary, step)
                return step, loss_, accuracy_

            def dev_step(batch_x, batch_y):
                feed_dict = {
                    instance.input_x: batch_x,
                    instance.input_y: batch_y}
                step, summary, loss_, accuracy_  = sess.run(
                    [
                        global_step,
                        merged_summary_op,                        
                        loss,
                        accuracy
                    ], feed_dict=feed_dict
                )

                dev_summary_writer.add_summary(summary, step)
                return step, loss_, accuracy_                

            for batch in generate():
                pass
            pass
        pass
