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
    how to update specific model -> tensorflow example GAN
    use multi-thread to process shared and private model simultaneously
    and test the function       
    Attributes:
        adv_loss (float): adversarial network loss 
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
        # TODO need to be tested
        self.task = tf.placeholder(tf.int32, name="task")

        self.private_model = []
        for task_name in params["task"]:
            with tf.variable_scope("{}-rnn".format(task_name)):
                rnn = RNN(sequence_length,
                          rnn_hidden_size,
                          private_num_layers,
                          dynamic=True,
                          use_attention=True,
                          attention_size=attention_size)
            self.private_model.append(rnn)  # TODO need to be tested            

        with tf.variable_scope("shared"):
            self.shared_model = RNN(sequence_length,
                                    rnn_hidden_size,
                                    shared_num_layers,
                                    dynamic=True,
                                    use_attention=True,
                                    attention_size=attention_size)

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
            s = self.shared_model.process(self.embedded_chars, seq_len)
            # batch_size, rnn_hidden_size * 2

        with tf.name_scope("private-model-processing"):
            # selected_model = tf.gather(self.private_model, self.task) # TODO
            private_outputs = []
            for model in self.private_model:
                output = model.process(self.embedded_chars, seq_len) # 
                # TODO ValueError: Variable bidirectional_rnn/fw/gru_cell/gates/kernel already exists, disallowed. \
                # Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? 
                private_outputs.append(output)
            # p = selected_model.process(self.embedded_chars, seq_len)
            p = tf.gather(private_outputs, task)
            # batch_size, rnn_hidden_size * 2

        with tf.name_scope("private-model-processing"):
            for i in range(len(params["task"])):
                p = tf.cond(tf.equal(task, i), lambda: self.private_model[i].process(self.embedded_chars, seq_len), lambda: 0) 
                               
               

        with tf.name_scope("discriminator-processing"):
            d = self.discriminator.process(s)
            # batch_size, num_tasks

        with tf.name_scope("loss"):
            sp = tf.concat([s, p], axis=1)
            # batch_size, rnn_hidden_size * 4
            adv_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=task_label, logits=d)
            self.adv_loss = tf.reduce_mean(adv_losses)
            self.diff_loss = tf.norm(
                tf.matmul(s, p, transpose_a=True), ord=2)  # TODO
            task_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y, logits=sp)
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
