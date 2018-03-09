import tensorflow as tf
import sys
sys.path.append("..")
import yaml
from model.rnn_model import RNN
from model.mlp_model import MLP

with open("../config/config.yaml", "r") as f:
    params = yaml.load(f)


class Adversarial_Network(object):
    """
    a batch of data come from the same task
    TODO
    trim the sequence to a fitted length according to specific task    
    use multi-thread to process shared and private model simultaneously 
    cf. https://arxiv.org/abs/1704.05742   
    Attributes:
        adv_loss (float): domain classification loss on adversarial network 
        task_loss (float): sentiment classification loss on specific task
        diff_loss (float): overlapping features between shared and private model
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 embedding_size,
                 vocab_size,
                 embedding_matrix,
                 static,
                 rnn_hidden_size,
                 shared_num_layers,
                 private_num_layers,
                 dynamic,
                 use_attention,
                 attention_size,
                 mlp_hidden_size):
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")
        self.task = tf.placeholder(tf.int32, name="task")

        self.rnn_model = RNN(sequence_length,
                             rnn_hidden_size,
                             private_num_layers,
                             dynamic=True,
                             use_attention=True,
                             attention_size=attention_size)

        # attempting to use uninitialized value beta2_power_2 if with tf.variable_scope("shared")
        # this is cause by Adam optimizer
        # if not in this with, it says no variables to optimize
        if embedding_matrix:
            self.W = tf.get_variable(shape=[vocab_size, embedding_size],
                                     initializer=tf.constant_initializer(
                embedding_matrix),
                name='W',
                trainable=not static)
        else:
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
        print("embedding matrix complete!")
        with tf.variable_scope("discriminator"):
            self.discriminator = MLP(sequence_length=rnn_hidden_size * 2,
                                     hidden_size=mlp_hidden_size,
                                     num_classes=len(params["task"]))

        task_label = tf.one_hot(self.task, len(params["task"]))
        task_label = tf.expand_dims(task_label, 0)
        batch_size = tf.shape(self.input_x)[0]
        task_label = tf.tile(task_label, multiples=[batch_size, 1])
        task_label = tf.cast(task_label, tf.float32)
        # batch_size, num_tasks

        with tf.name_scope("embedding-layer"):
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("sequence-length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("shared-model-processing"):
            s = self.rnn_model.process(
                self.embedded_chars, seq_len, scope="shared")
            # batch_size, rnn_hidden_size * 2

        # with tf.name_scope("private-model-processing"):
        #     # selected_model = tf.gather(self.private_model, self.task) # didn't work
        #     private_outputs = []
        #     for model in self.private_model:
        #         output = model.process(self.embedded_chars, seq_len)
        #         # TODO ValueError: Variable bidirectional_rnn/fw/gru_cell/gates/kernel already exists, disallowed. \
        #         # Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
        #         private_outputs.append(output)
        #     # p = selected_model.process(self.embedded_chars, seq_len)
        #     p = tf.gather(private_outputs, task)
        #     # batch_size, rnn_hidden_size * 2

        with tf.name_scope("private-model-processing"):
            useless = tf.constant(
                [0]*2*rnn_hidden_size, dtype=tf.float32)
            useless = tf.expand_dims(useless, 0)
            useless = tf.tile(useless, multiples=[batch_size, 1])
            # shape of all inputs of op gather must match

            def fn(i):
                output = self.rnn_model.process(
                    self.embedded_chars, seq_len, "private-{}".format(params["task"][i]))
                return output
            l = []
            for i in range(len(params["task"])):
                temp = tf.cond(tf.equal(self.task, i),
                               lambda: fn(i), lambda: useless)
                # set reuse=True or reuse=tf.AUTO_REUSE
                l.append(temp)
            p = tf.gather(l, self.task)
            # batch_size, rnn_hidden_size * 2

        with tf.name_scope("discriminator-processing"):
            d = self.discriminator.process(s)
            # batch_size, num_tasks

        with tf.name_scope("fully-connected-layer"):
            sp = tf.concat([s, p], axis=1)
            # batch_size, rnn_hidden_size * 4
            w = tf.Variable(tf.truncated_normal(
                [rnn_hidden_size*4, num_classes], stddev=0.1), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(sp, w, b)

        with tf.name_scope("loss"):
            adv_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=task_label, logits=d)
            self.adv_loss = tf.reduce_mean(adv_losses)
            diff_losses = tf.norm(
                tf.multiply(s, p), ord=2, axis=1)  # TODO still need to be tested
            self.diff_loss = tf.reduce_mean(diff_losses)
            task_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y, logits=scores)  # logits and labels must be same size
            self.task_loss = tf.reduce_mean(task_losses)

        with tf.name_scope("task-accuracy"):
            predictions = tf.argmax(scores, 1, name="predictions")
            correct_predictions = tf.equal(
                predictions, tf.argmax(self.input_y, 1))
            self.task_accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("discriminator-accuracy"):
            predictions = tf.argmax(d, 1, name="predictions")
            correct_predictions = tf.equal(
                predictions, tf.argmax(task_label, 1))
            self.discriminator_accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
