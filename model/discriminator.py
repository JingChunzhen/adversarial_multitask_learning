import tensorflow as tf


class Discriminator():

    def __init__(self):
        '''
        labels: task label 
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name='X')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='Y')

        with tf.name_scope("fully-connected-layer"):
            W = tf.Variable(tf.truncated_normal(
                [sequence_length, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.input_x, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
