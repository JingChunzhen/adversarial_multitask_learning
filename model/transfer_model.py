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
                 attention_size,
                 embedding_matrix,
                 fc_w,
                 fc_b):
        """
        using embedding_matrix, shared-model and fully-conneted params to initialize the transfer model 
        Args:
            embedding_matrix (list): embedding matrix of pre-trained adversarial model  
            fc_w (list): fully connected weight of pre-trained model
            fc_b (list): fully connected bias of pre-trained model         
        """
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")

        self.rnn_model = RNN(sequence_length,
                             rnn_hidden_size,
                             num_layers,
                             dynamic=True,
                             use_attention=True,
                             attention_size=attention_size)

        self.W = tf.get_variable(shape=[vocab_size, embedding_size],
                                 initializer=tf.constant_initializer(
                                     embedding_matrix),
                                 name='W',
                                 trainable=not static)

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
            TODO
            initialize the rnn model using pre-trained adversarial model 
            """
            s = self.rnn_model.process(
                self.embedded_chars, seq_len, scope="transfer-shared")

        with tf.name_scope("fully-connected-layer"):
            w = tf.get_variable(shape=[rnn_hidden_size*2, num_classes],
                                initializer=tf.constant_initializer(fc_w),
                                name='w')
            b = tf.get_variable(shape=[num_classes],
                                initializer=tf.constant_initializer(fc_b),
                                name='b')
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