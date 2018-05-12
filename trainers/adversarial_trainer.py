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
from utils.data_loader import load_data, batch_iter
from itertools import chain

with open("../config/config.yaml", "rb") as f:
    params = yaml.load(f)


class EVAL(object):
    """
    # TODO add visualization of training process 
    Adversarial Network Training    
    """

    def __init__(self, sequence_length):
        # load data first
        self.processor = learn.preprocessing.VocabularyProcessor.restore(
            "../temp/vocab")
        self.processor.max_document_length = sequence_length
        raw_data, raw_label = load_data("train")
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

        self.test_data = []
        self.test_label = []
        raw_data, raw_label = load_data("test")
        for rd, rl in zip(raw_data, raw_label):
            tmp_data = []
            tmp_label = []
            rd = list(self.processor.transform(rd))
            for tmp_x, tmp_y in zip(rd, rl):
                tmp_x = tmp_x.tolist()
                if np.sum(tmp_x) != 0:
                    tmp_data.append(tmp_x)
                    tmp_label.append(tmp_y)
            self.test_data.append(tmp_data)
            self.test_label.append(tmp_label)
        del raw_data, raw_label
        print("load test data complete!")
        self.embedding_matrix = self._embedding_matrix_initializer() if os.path.exists(
            "../data/glove.6B/glove.6B.{}d.txt".format(params["global"]["embedding_size"])) else None
        print("read from embedding_matrix complete!")

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

    def process(self, learning_rate, batch_size, epochs, lamda, evaluate_every, save_every):
        """
        """
        embedding_matrix = list(chain.from_iterable(
            self.embedding_matrix)) if self.embedding_matrix else None
        with tf.Graph().as_default():
            instance = Adversarial_Network(
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

            global_step = tf.Variable(0, trainable=False)

            # adv_loss = instance.adv_loss
            disc_loss = instance.disc_loss
            gen_loss = instance.gen_loss
            diff_loss = instance.diff_loss
            task_loss = instance.task_loss

            discriminator_accuracy = instance.discriminator_accuracy
            # TODO to jointly train the discriminator using output of private domain
            task_accuracy = instance.task_accuracy

            discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate)
            task_optimizer = tf.train.AdamOptimizer(learning_rate)
            shared_optimizer = tf.train.AdamOptimizer(learning_rate)
            # use AdamOptimizer may cause error
            # https://github.com/amitmac/Question-Answering/issues/2

            discriminator_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            shared_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared")

            task_train_op = task_optimizer.minimize(
                task_loss + lamda * diff_loss, global_step=global_step)
            discriminator_train_op = discriminator_optimizer.minimize(
                disc_loss, var_list=discriminator_vars)
            shared_train_op = shared_optimizer.minimize(
                -1 * 0.05 * gen_loss, var_list=shared_vars)  # No varibales to optimize

            tf.summary.scalar("disc_loss", disc_loss)
            tf.summary.scalar("gen_loss", gen_loss)
            tf.summary.scalar("diff_loss", diff_loss)
            tf.summary.scalar("task_loss", task_loss)
            tf.summary.scalar("task_accuracy", task_accuracy)
            tf.summary.scalar("discriminator_accuracy", discriminator_accuracy)

            merged_summary_op = tf.summary.merge_all()

            init = tf.global_variables_initializer()  # this line of code must be here
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init)

                train_summary_writer = tf.summary.FileWriter(
                    logdir='../temp/summary/adversarial/train', graph=sess.graph)
                dev_summary_writer = tf.summary.FileWriter(
                    logdir='../temp/summary/adversarial/dev', graph=sess.graph)

                def train_step(task, x_batch, y_batch):
                    feed_dict = {
                        instance.task: task,
                        instance.input_x: x_batch,
                        instance.input_y: y_batch,
                        instance.input_keep_prob: params["global"]["input_keep_prob"],
                        instance.output_keep_prob: params["global"]["output_keep_prob"]
                    }
                    step, summary, _, _, _, diff_loss_, disc_loss_, gen_loss_, task_loss_, dis_acc_, task_acc_ = sess.run(
                        [
                            global_step,
                            merged_summary_op,
                            discriminator_train_op,
                            shared_train_op,
                            task_train_op,
                            diff_loss,
                            disc_loss,
                            gen_loss,
                            task_loss,
                            discriminator_accuracy,
                            task_accuracy
                        ], feed_dict=feed_dict
                    )

                    train_summary_writer.add_summary(summary, step)
                    return step, diff_loss_, disc_loss_, gen_loss_, task_loss_, dis_acc_, task_acc_

                def dev_step(task, x_batch, y_batch):
                    feed_dict = {
                        instance.task: task,
                        instance.input_x: x_batch,
                        instance.input_y: y_batch,
                        instance.input_keep_prob: 1.0,
                        instance.output_keep_prob: 1.0
                    }
                    step, summary, diff_loss_, disc_loss_, gen_loss_, task_loss_, dis_acc_, task_acc_ = sess.run(
                        [
                            global_step,
                            merged_summary_op,
                            diff_loss,
                            disc_loss,
                            gen_loss,
                            task_loss,
                            discriminator_accuracy,
                            task_accuracy
                        ], feed_dict=feed_dict
                    )
                    dev_summary_writer.add_summary(summary, step)
                    return step, diff_loss_, disc_loss_, gen_loss_, task_loss_, dis_acc_, task_acc_

                for task, batch in batch_iter(self.train_data, self.train_label, batch_size, epochs, shuffle=False):
                    x_batch, y_batch = zip(*batch)
                    current_step, diff_loss_, disc_loss_, gen_loss_, task_loss_, dis_acc_, task_acc_ = train_step(
                        task, x_batch, y_batch)

                    print("step: {}, discriminator loss: {:.5f}, generator loss: {:.5f}, diff loss: {:.5f}, task loss: {:.5f}, discriminator accuracy: {:.2f}, task accuracy: {:.2f}".format(
                        current_step,
                        disc_loss_,
                        gen_loss_,
                        diff_loss_,
                        task_loss_,
                        dis_acc_,
                        task_acc_))

                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % evaluate_every == 0:
                        print("Evaluation:")
                        diff_losses = []
                        disc_losses = []
                        gen_losses = []
                        task_losses = []
                        dis_accuracies = []
                        task_accuracies = []

                        cnt = 0
                        for task, batch in batch_iter(self.test_data, self.test_label, 50, 1, shuffle=True):
                            x_dev, y_dev = zip(*batch)
                            _, diff_loss_, disc_loss_, gen_loss_, task_loss_, dis_acc_, task_acc_ = dev_step(
                                task, x_dev, y_dev)

                            diff_losses.append(diff_loss_)
                            disc_losses.append(disc_loss_)
                            gen_losses.append(gen_loss_)
                            task_losses.append(task_loss_)
                            dis_accuracies.append(dis_acc_)
                            task_accuracies.append(task_acc_)

                            cnt += 1
                            if cnt == 10:
                                break

                        print("discriminator loss: {:.5f}, generator loss: {:.5f}, diff loss: {:.5f}, task loss: {:.5f}, discriminator accuracy: {:.2f}, task accuracy: {:.2f}".format(
                            np.mean(disc_losses),
                            np.mean(gen_losses),
                            np.mean(diff_losses),
                            np.mean(task_losses),
                            np.mean(dis_accuracies),
                            np.mean(task_accuracies)))

                    if current_step % save_every == 0:
                        saver.save(sess, "../temp/model/adversarial/model",
                                   global_step=current_step)


if __name__ == "__main__":
    eval = EVAL(params["global"]["sequence_length"])

    eval.process(
        learning_rate=params["global"]["learning_rate"],
        batch_size=params["global"]["batch_size"],
        epochs=params["global"]["epochs"],
        lamda=params["global"]["lamda"],
        evaluate_every=40,
        save_every=500
    )
