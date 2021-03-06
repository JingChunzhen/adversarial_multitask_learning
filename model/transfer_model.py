import tensorflow as tf
import sys
sys.path.append("..")
import yaml
from model.rnn_model import RNN
from model.mlp_model import MLP

with open("../config/config.yaml", "r") as f:
    params = yaml.load(f)


class Transfer(object):
    """
    transfer learning using shared model in adversarial network
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 embedding_size,
                 vocab_size,
                 static,
                 rnn_hidden_size,
                 num_layers,
                 dynamic,
                 use_attention,
                 attention_size):
        """
        transfer model contains embedding layer, rnn layer and fully-connected layer
        and will all be initialized by the corresponding params of adversarial network
        the rnn params will be initialized by the shared rnn model in adversarial network           
        """
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")
        self.input_keep_prob = tf.placeholder(tf.float32, name="keep_prob_in")
        self.output_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_out")

        self.rnn_model = RNN(sequence_length,
                             rnn_hidden_size,
                             num_layers,
                             dynamic=True,
                             use_attention=True,
                             attention_size=attention_size)

        self.W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="transfer-W")

        with tf.name_scope("embedding-layer"):
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("sequence-length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("rnn-processing"):
            """            
            initialize the rnn model using pre-trained adversarial model 
            """
            s = self.rnn_model.process(
                self.embedded_chars, seq_len, self.input_keep_prob, self.output_keep_prob, scope="transfer-shared", )

        with tf.name_scope("transfer-fully-connected-layer"):
            w = tf.Variable(tf.truncated_normal(
                [rnn_hidden_size*2, num_classes], stddev=0.1), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(s, w, b)

        with tf.name_scope("loss"):
            task_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y, logits=scores)  # logits and labels must be same size
            self.task_loss = tf.reduce_mean(task_losses)

        with tf.name_scope("task-accuracy"):
            self.predictions = tf.argmax(scores, 1, name="predictions")
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.task_accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
