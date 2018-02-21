import tensorflow as tf
import yaml
from lstm_model import RNN
from discriminator import MLP

with open("../config/config.yaml", "r") as f:
    params = yaml.load(f)


class Adversial_Network(object):
    """
    a batch of data come from the same task
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 embedding_size,
                 vocab_size,
                 embedding_matrix,
                 static,
                 rnn_hidden_size,
                 rnn_num_layers,
                 dynamic,
                 use_attention,
                 attention_size,
                 mlp_hidden_size):
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")

        self.private_model = []
        for task in params["task"]:
            with tf.variable_scope("{}-rnn".format(task)):
                rnn = RNN(sequence_length,
                          rnn_hidden_size,
                          rnn_num_layers,
                          dynamic=True,
                          use_attention=True,
                          attention_size=attention_size)
            self.private_model.append(rnn)  # TODO

        with tf.variable_scope("shared"):
            self.shared_model = RNN(sequence_length,
                                    rnn_hidden_size,
                                    rnn_num_layers,
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
            self.discriminator = MLP(sequence_length=rnn_hidden_size*4,
                                     hidden_size=mlp_hidden_size,
                                     num_classes=len(params["task"]))                                    
        self.sequence_length = sequence_length

    def process(self, task):
        """
        Args:
            task (int): task indice
        Returns:
            adv_loss (float): adversarial network loss 
            task_loss (float): sentiment classification loss on specific task
            diff_loss (float): overlapping features between shared and private model
        """
        # TODO
        # trim the sequence to a fitted length according to specific task
        # how to update specific model -> tensorflow example GAN
        # use multi-thread to process shared and private model simultaneously
        # and test the function

        with tf.name_scope("embedding-layer"):
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("sequence-length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=self.sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("shared-model-processing"):
            s = self.shared_model.process(self.embedded_chars, seq_len)

        with tf.name_scope("private-model-processing"):
            p = self.private_model[task].process(self.embedded_chars, seq_len)        

        with tf.name_scope("fully-connected-layer"):
            sp = tf.concat([s, p], axis=0)
            W = tf.Variable(tf.truncated_normal(
                [lstm_hidden_size * 2, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(
                sp, W, b, name="scores")  # TODO error
            predictions = tf.argmax(scores, 1, name="predictions")

        with tf.name_scope("loss"):
            adv_losses = self.discriminator.loss
            adv_loss = tf.reduce_mean(adv_losses)
            diff_loss = tf.norm(tf.matmul(s, p, transpose_a=True), ord=2) # TODO
            task_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y, logits=scores)
            task_loss = tf.reduce_mean(task_losses)

        with tf.name_scope("accuracy"):        
            correct_predictions = tf.equal(
                predictions, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
        
        return adv_loss, diff_loss, task_loss, accuracy
