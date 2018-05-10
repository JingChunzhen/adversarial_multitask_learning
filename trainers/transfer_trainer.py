import os
import sys
sys.path.append("..")
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from model.adversarial_model import Adversarial_Network
from model.transfer_model import Transfer
from utils.data_loader import load_data_v2, batch_iter_v2
from itertools import chain

with open("../config/config.yaml", "rb") as f:
    params = yaml.load(f)


class EVAL(object):
    """
    transfer learning using shared model in adversarial network
    """

    def __init__(self):
        """data preparation 
        """
        raw_x, raw_y = load_data_v2()
        self.max_document_length = 32  # this is optimal length of twitter airline

        try:
            self.processor = learn.preprocessing.VocabularyProcessor.restore(
                "../temp/vocab")
            raw_x = list(self.processor.transform(raw_x))

            x, y = [], []
            for tmp_x, tmp_y in zip(raw_x, raw_y):
                tmp_x = tmp_x.tolist()
                if np.sum(tmp_x) != 0:
                    x.append(tmp_x)  # rid x with all 0s
                    y.append(tmp_y)

            x_temp, self.x_test, y_temp, self.y_test = train_test_split(
                x, y, test_size=params["transfer"]["test_size"])
            self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(
                x_temp, y_temp, test_size=params["transfer"]["validate_size"])

            del x_temp, y_temp, raw_x, x, y
        except IOError as e:
            print("File error {}".format(e))
        self.instance = None

    def _params_initializer(self, model_path):
        """
        initialize parameters of the transfer model using pre-trained adversarial network 
        this is supposed to be used in trainers 
        Args:
            model_path (string): path stored pre-trained adversarial model 
        Returns:
            embedding_matrix (list)
            fc_params: fully-connected weights and bias 
            shared_model_vars (list of matrix): shared rnn model
        """
        adn = Adversarial_Network(
            sequence_length=params["global"]["sequence_length"],
            num_classes=params["global"]["num_classes"],
            embedding_size=params["global"]["embedding_size"],
            vocab_size=len(
                self.processor.vocabulary_),
            embedding_matrix=None,
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
        fc_w = fc_w[:params["global"]["rnn_hidden_size"] * 2]
        fc_w = list(chain.from_iterable(fc_w))
        fc_b = list(chain.from_iterable(fc_b))
        return embedding_matrix, fc_w, fc_b, shared_model_vars

    def _rnn_initializer(self, shared_model_vars):
        """
        TODO: need to be tested 
        initialize the rnn model in transfer model using pre-trained adversarial network
        Args:            
            shared_model_vars (list): pre-trained shared model's variables 
        """
        rnn_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="transfer-shared")
        for rnn_var, shared_model_var in zip(rnn_vars, shared_model_vars):
            tf.assign(rnn_var, shared_model_var)

    def process(self, model_path, learning_rate, batch_size, epochs, evaluate_every):
        """
        incremental learning based on the pre-trained adversarial model 
        Args:
            model_path (string): path stored the pre-trained adversarial network
            batch_size (int): batch_size in training process         
        """
        embedding_matrix, fc_w, fc_b, shared_model_vars = self._params_initializer(
            model_path)
        instance = Transfer(
            sequence_length=params["global"]["sequence_length"],
            num_classes=params["global"]["num_classes"],
            embedding_size=params["global"]["embedding_size"],
            vacab_size=len(embedding_matrix) /
            params["global"]["embedding_size"],
            static=params["global"]["static"],
            rnn_hidden_size=params["global"]["rnn_hidden_size"],
            num_layers=params["shared"]["num_layers"],
            dynamic=params["global"]["dynamic"],
            use_attention=params["global"]["use_attention"],
            attention_size=params["global"]["attention_size"],
            embedding_matrix=embedding_matrix,
            fc_w=fc_w,
            fc_b=fc_b)

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
                pred_, summary, step, loss_, accuracy_ = sess.run(
                    [
                        preds,
                        merged_summary_op,
                        global_step,
                        loss,
                        accuracy
                    ], feed_dict=feed_dict
                )

                dev_summary_writer.add_summary(summary, step)
                return pred_, loss_, accuracy_

            for batch in batch_iter_v2(list(zip(self.x_train, self.y_train)), batch_size, epochs):
                x_batch, y_batch = zip(*batch)
                current_step, loss_, accuracy_ = train_step(
                    x_batch, y_batch)
                print("Training, step: {}, accuracy: {:.2f}, loss: {:.5f}".format(
                    current_step, accuracy_, loss_))
                # current_step = tf.train.global_step(sess, global_step)

                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")

                    losses = []
                    accuracies = []

                    y_true = []
                    y_pred = []

                    for batch in batch_iter_v2(list(zip(self.x_validate, self.y_validate)), 50, 1):
                        if random.randint(0, 3) != 0:
                            continue
                        x_dev, y_dev = zip(*batch)
                        pred_, loss_, accuracy_ = dev_step(x_dev, y_dev)
                        accuracies.append(accuracy_)
                        losses.append(loss_)

                        y_pred.extend(pred_.tolist())
                        y_true.extend(np.argmax(y_dev, axis=1).tolist())

                    print("Evaluation Accuracy: {}, Loss: {}".format(
                        np.mean(accuracies), np.mean(losses)))
                    print(classification_report(
                        y_true=y_true, y_pred=y_pred))


if __name__ == "__main__":
    transfer = EVAL()
    transfer.process(
        model_path="../temp/model",
        learning_rate=0.0001,
        batch_size=128,
        epochs=100
    )
    pass
