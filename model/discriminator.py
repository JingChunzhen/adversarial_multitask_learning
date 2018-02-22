import tensorflow as tf


class MLP(object):

    def __init__(self, sequence_length, num_classes, hidden_size, l2_reg_lambda):
        '''
        labels: task label 
        '''        
        self.l2_loss = tf.constant(0.0)

        self.W1 = tf.Variable(tf.truncated_normal(
            [sequence_length, hidden_size], stddev=0.1), name="W1")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b1")

        self.W2 = tf.Variable(tf.truncated_normal(
            [hidden_size, num_classes], stddev=0.1), name="W2")
        self.b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
        self.l2_reg_lambda = l2_reg_lambda

    def process(self, x)
        """
        Args:
            x (tensor): the shared model output with shape (batch_size, sequence_length)
                sequence_length == rnn_hidden_size * 2
            y (tensor): shape: (batch_size, num_classes)
                num_classes == number of tasks
        Returns:
            loss (float), accuracy (float)
        """
        with tf.name_scope("fully-connected-layer1"):            
            self.l2_loss += tf.nn.l2_loss(self.W1)
            self.l2_loss += tf.nn.l2_loss(self.b1)
            scores = tf.nn.xw_plus_b(
                x, self.W1, self.b1, name="scores1")

        with tf.name_scope("fully-connected-layer2"):
            self.l2_loss += tf.nn.l2_loss(self.W2)
            self.l2_loss += tf.nn.l2_loss(self.b2)
            scores = tf.nn.xw_plus_b(scores, self.W2, self.b2, name="scores2")
            # predictions = tf.argmax(scores, 1, name="predictions")
        return scores
        
        """
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                predictions, tf.argmax(y, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
        return loss, accuracy
        """