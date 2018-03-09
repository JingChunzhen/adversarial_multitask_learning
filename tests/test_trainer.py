import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append("..")
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from model.adversarial_model import Adversarial_Network
from utils.data_loader import load_data, batch_iter
from itertools import chain

with open("../config/config.yaml", "rb") as f:
    params = yaml.load(f)


class EVAL(object):
    """    
    This is a training without adversarial network    
    """

    def __init__(self, sequence_length):
        # load data first
        self.processor = learn.preprocessing.VocabularyProcessor.restore(
            "../temp/vocab")
        self.processor.max_document_length = sequence_length
        raw_data, raw_label = load_data("test")
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
        
    def process(self, learning_rate, batch_size, epochs, evaluate_every):
        """
        """       
        with tf.Graph().as_default():
            instance = Adversarial_Network(
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

            global_step = tf.Variable(0, trainable=False)
            init = tf.global_variables_initializer()

            # discriminator_optimizer = tf.train.AdamOptimizer(learning_rate)
            # task_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # shared_optimizer = tf.train.AdamOptimizer(learning_rate)
            advloss = instance.adv_loss
            taskloss = instance.task_loss

            # discriminator_vars = tf.get_collection(
            #     tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator") # OK 
            # print(discriminator_vars)
            shared_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared")
            print(shared_vars)
            apparel_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="apparel-rnn")
            print(apparel_vars)
            books_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="books-rnn")
            print(books_vars)
       
            with tf.Session() as sess:
                sess.run(init)
                # dvs = sess.run(discriminator_vars)
                # print(dvs)
                svs = sess.run(shared_vars)
                print(svs)
                avs = sess.run(apparel_vars)
                print(avs)
                bvs = sess.run(books_vars)
                print(bvs)

                for task, batch in batch_iter(self.train_data, self.train_label, batch_size, epochs, shuffle=False):
                    x, y = zip(*batch)
                    al, tl, svs = sess.run([advloss, taskloss, shared_vars], feed_dict={instance.task: task, instance.input_x: x, instance.input_y: y})
                    print(al)
                    print(tl)
                    print(svs)
                    
                    

if __name__ == "__main__":
    # https://stackoverflow.com/questions/45263666/tensorflow-variable-reuse
    # https://sthsf.github.io/2017/06/18/ValueError:%20kernel%20already%20exists/index.html
    # https://stackoverflow.com/questions/35013080/tensorflow-how-to-get-all-variables-from-rnn-cell-basiclstm-rnn-cell-multirnn
    # 必须显示的将rnn的运行过程也定义在variable_scope中才可以进行下去
    # 而且就目前来看，只需要知道shared和disscriminator的variable_scope
    eval = EVAL(params["global"]["sequence_length"])
    eval.process(
        learning_rate=params["global"]["learning_rate"],
        batch_size=params["global"]["batch_size"],
        epochs=params["global"]["epochs"],
        evaluate_every=100
    )