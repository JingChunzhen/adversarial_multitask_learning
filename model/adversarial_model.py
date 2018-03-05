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
        self.task = tf.placeholder(tf.int32, name="task") # TODO need to be tested 

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
            self.discriminator = MLP(sequence_length=rnn_hidden_size*2,
                                     hidden_size=mlp_hidden_size,
                                     num_classes=len(params["task"]))                                    
        self.sequence_length = sequence_length
        self.adv_loss = None
        self.diff_loss = None
        self.task_loss = None
        self.task_accuracy = None
        self.discriminator_accuracy = None
        #self.task = None
    
    def task_indice(self, task_indice):
        """
        get int type indice 
        """
        feed_dict = {self.task: task_indice}
        with tf.Session() as sess:
            task_indice = sess.run(task, feed_dict=feed_dict)
        return task_indice        

    def process(self, task):
        """
        Arg:
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
        l = [0] * len(params["task"])
        l[task] = 1
        task_label = tf.constant(l, dtype=tf.float32)
        task_label = tf.expand_dims(task_label, 0)
        batch_size = tf.shape(self.input_x)[0]
        task_label = tf.tile(task_label, multiples=[batch_size, 1])
        # batch_size, num_tasks

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
            # batch_size, rnn_hidden_size * 2

        with tf.name_scope("private-model-processing"):
            p = self.private_model[task].process(self.embedded_chars, seq_len)        
            # batch_size, rnn_hidden_size * 2
        
        with tf.name_scope("discriminator-processing"):
            d = self.discriminator.process(s)
            # batch_size, num_tasks

        # with tf.name_scope("fully-connected-layer"):
        #     W = tf.Variable(tf.truncated_normal(
        #         [rnn_hidden_size * 2, num_classes], stddev=0.1), name="W")
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     scores = tf.nn.xw_plus_b(
        #         sp, W, b, name="scores")               

        with tf.name_scope("loss"):
            sp = tf.concat([s, p], axis=1)  
            # batch_size, rnn_hidden_size * 4
            adv_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=task_label, logits=d)
            self.adv_loss = tf.reduce_mean(adv_losses)
            self.diff_loss = tf.norm(tf.matmul(s, p, transpose_a=True), ord=2) # TODO
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
                predictions, tf.cast(task_label, tf.int32))
            self.discriminator_accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
        
