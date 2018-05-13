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


###
# 测试不同的tensorflow model下使用get_collections根据scope获取变量的时候，会发生什么事情
###
class EVAL(object):
    """
    transfer learning using shared model in adversarial network
    """

    def __init__(self):
        """data preparation 
        """
        raw_x, raw_y = load_data_v2()
        print(">>> raw data")
        print(np.shape(raw_x))
        # self.max_document_length = 800  # this is optimal length of twitter airline

        try:
            self.processor = learn.preprocessing.VocabularyProcessor.restore(
                "../temp/vocab")
            self.processor.max_document_length = 800
            raw_x = list(self.processor.transform(raw_x))
            print(np.shape(raw_x))  # -> 11541, 340

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
        print(">>> data")
        print(np.shape(self.x_train))
        print(np.shape(self.y_train))
        self.source_model = None
        self.target_model = None

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
        self.source_model = Adversarial_Network(
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
        # shared_model_vars = tf.get_collection(
        #     tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")
        fc_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="fully-connected-layer")

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_path)
            # sess.run(ops) # assign params from source to target
            embedding_matrix = sess.run(embedding_vars)
            fc_params = sess.run(fc_vars)
            # shared_model_vars = sess.run(shared_model_vars)

        print(">>> embedding matrix")
        print(np.shape(embedding_matrix))
        embedding_matrix = list(chain.from_iterable(embedding_matrix[0]))
        print(">>> embedding matrix")
        print(np.shape(embedding_matrix))  # (687700,)
        fc_w, fc_b = fc_params
        fc_w = fc_w[:params["global"]["rnn_hidden_size"] * 2]
        fc_w = list(chain.from_iterable(fc_w))
        print(">>>> fc_w")
        # print(fc_w)
        print(np.shape(fc_w))  # 2 * rnn_hidden_size -> 512
        print(">>> fc_b")
        print(np.shape(fc_b))  # 2
        # fc_b = list(chain.from_iterable(fc_b)) #
        return embedding_matrix, fc_w, fc_b

    def _rnn_initializer(self, sess):
        """
        copy params from source to target
        initialize the rnn model in transfer model using pre-trained adversarial network
        Args:
            sess: tensoflow session
        Return:
            ops (list of assignment)             
        """
        ops = []
        source_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared")
        print(source_vars)
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="transfer-shared")
        print(target_vars)
        for target_var, source_var in zip(target_vars, source_vars):
            print(target_var)
            print(source_var)

        for target_var, source_var in zip(target_vars, source_vars):
            op = target_var.assign(source_var)
            ops.append(op)

        sess.run(ops)

    def _get_source_vars(self, model_path):
        """   
        Args:
            model_path (string): path stored the pre-trained adversarial net work        
        """
        self.source_model = Adversarial_Network(
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

        target_embeddings = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="transfer-W")
        print(target_embeddings) # -> []
        target_rnn_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="transfer-shared")       
        print(target_rnn_vars) 
        target_fc_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="transfer-fully-connected-layer")

        source_embeddings = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="W")
        print(source_embeddings)
        source_rnn_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")
        source_fc_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="fully-connected-layer")

        ops = []
        op = target_embeddings.assign(source_embeddings)
        ops.append(op)
        for target_rnn_var, source_rnn_var in zip(target_rnn_vars, source_rnn_vars):
            op = target_rnn_var.assign(source_rnn_var)
            ops.append(op)

        target_fc_w, target_fc_b = target_fc_vars
        source_fc_w, source_fc_b = source_fc_vars
        target_fc_w.assign(source_fc_w[: 2*params["global"]["rnn_hidden_size"]])
        target_fc_b.assign(source_fc_b)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_path)            
            sess.run(ops)



    def process(self, model_path, learning_rate, batch_size, epochs, evaluate_every):
        """
        incremental learning based on the pre-trained adversarial model 
        Args:
            model_path (string): path stored the pre-trained adversarial network
            batch_size (int): batch_size in training process         
        """
        embedding_matrix, fc_w, fc_b = self._params_initializer(model_path)
        instance = Transfer(
            sequence_length=params["global"]["sequence_length"],
            num_classes=params["global"]["num_classes"],
            embedding_size=params["global"]["embedding_size"],
            vocab_size=len(
                self.processor.vocabulary_),
            static=params["global"]["static"],
            rnn_hidden_size=params["global"]["rnn_hidden_size"],
            num_layers=params["shared_model"]["num_layers"],
            dynamic=params["global"]["dynamic"],
            use_attention=params["global"]["use_attention"],
            attention_size=params["global"]["attention_size"],
            embedding_matrix=embedding_matrix,
            fc_w=fc_w,
            fc_b=fc_b)
        # self._rnn_initializer()

        global_step = tf.Variable(0, trainable=False)
        loss = instance.task_loss
        accuracy = instance.task_accuracy
        preds = instance.predictions

        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("loss", loss)

        merged_summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            #self._params_initializer(model_path)
            self._rnn_initializer(sess)
            
            train_summary_writer = tf.summary.FileWriter(
                logdir='../temp/summary/transfer/train', graph=sess.graph)
            dev_summary_writer = tf.summary.FileWriter(
                logdir='../temp/summary/transfer/dev', graph=sess.graph)

            def train_step(batch_x, batch_y):
                feed_dict = {
                    instance.input_x: batch_x,
                    instance.input_y: batch_y,
                    instance.input_keep_prob: 0.75,
                    instance.output_keep_prob: 0.75}
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
                    instance.input_y: batch_y,
                    instance.input_keep_prob: 1.0,
                    instance.output_keep_prob: 1.0}
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
        model_path="../temp/model/adversarial/model-500",
        learning_rate=0.0001,
        batch_size=128,
        epochs=100,
        evaluate_every=40
    )
    pass
#"../temp/model/adversarial/model"


"""
model 1
model 2
assign op

sess.run (init)

init model 1

sess.run (op)
"""