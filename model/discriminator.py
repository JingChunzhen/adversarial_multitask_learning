import tensorflow as tf


class MLP():

    def __init__(self, sequence_length, num_classes, hidden_size):
        '''
        labels: task label 
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name='X')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='Y')
        l2_loss = tf.constant(0.0)

        with tf.name_scope("fully-connected-layer1"):
            W = tf.Variable(tf.truncated_normal(
                [sequence_length, hidden_size], stddev=0.1), name="W1")
            b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b1")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.input_x, W, b, name="scores1")

        with tf.name_scope("fully-connected-layer2"):
            W = tf.Variable(tf.truncated_normal(
                [hidden_size, num_classes], stddev=0.1), name="W2")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            self.scores = tf.nn.xw_plus_b(self.scores, W, b, name="scores2")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
