import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from rnn.attention import attention


class RNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, num_layers, l2_reg_lambda, dynamic, use_attention,
                 attention_size, embedding_init, embedding_matrix, static):
        '''
        # TODO: dropout layer before fc layer 
        Args:
            sequence_length (int)
            num_classes (int):
            vocab_size (int):
            embedding_size (int):            
            hidden_size (int):
            num_layer (int):
            l2_reg_lambda (float):
            dynamic (boolean):
            use_attention (boolean):
            attention_size (int):
            embedding_init (boolean): True for initialize the embedding layer with glove false for not 
            embedding_matrix (list of float): length vocabulary size * embedding_size
            static (boolean): False for embedding_layer trainable during training false True for not 
        '''
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="x")
        # convert to dtype: list(list) in case the error in tf.sign
        # original dtype list(np.ndarray)
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="y")  # dtype: list(list)

        self.input_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_in")
        self.output_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_out")

        l2_loss = tf.constant(0.0)

        if embedding_init:
            with tf.name_scope("embedding-layer-with-glove-initialized"):
                self.W = tf.get_variable(shape=[vocab_size, embedding_size], initializer=tf.constant_initializer(
                    embedding_matrix), name='W', trainable=not static)
                # initializer can only be constant value or list with N-dimension
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
        else:
            with tf.name_scope("embedding-layer"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)

        self.embedded_chars = tf.unstack(
            self.embedded_chars, sequence_length, axis=1)
        # get list (length == sequence_length) of tensors with shape: batch_size, embedding_size        

        with tf.name_scope("sequence-length"):
            mask = tf.sign(self.input_x)
            range_ = tf.range(
                start=1, limit=sequence_length + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")  # element wise
            self.seq_len = tf.reduce_max(mask, axis=1)

        with tf.name_scope("forward-cell"):
            if num_layers != 1:
                cells = []
                for i in range(num_layers):
                    rnn_cell = DropoutWrapper(
                        GRUCell(hidden_size),
                        input_keep_prob=self.input_keep_prob,
                        output_keep_prob=self.output_keep_prob
                    )
                    cells.append(rnn_cell)
                self.cell_fw = MultiRNNCell(cells)
            else:
                self.cell_fw = DropoutWrapper(
                    GRUCell(hidden_size),
                    input_keep_prob=self.input_keep_prob,
                    output_keep_prob=self.output_keep_prob
                )

        with tf.name_scope("backward-cell"):
            if num_layers != 1:
                cells = []
                for i in range(num_layers):
                    rnn_cell = DropoutWrapper(
                        GRUCell(hidden_size),
                        input_keep_prob=self.input_keep_prob,
                        output_keep_prob=self.output_keep_prob
                    )
                    cells.append(rnn_cell)
                self.cell_bw = MultiRNNCell(cells)
            else:
                self.cell_bw = DropoutWrapper(
                    GRUCell(hidden_size),
                    input_keep_prob=self.input_keep_prob,
                    output_keep_prob=self.output_keep_prob
                )

        if dynamic:
            with tf.name_scope("dynamic-rnn-with-{}-layers".format(num_layers)):
                outputs, _, _ = tf.nn.static_bidirectional_rnn(
                    inputs=self.embedded_chars,
                    cell_fw=self.cell_fw,
                    cell_bw=self.cell_bw,
                    sequence_length=self.seq_len,
                    dtype=tf.float32
                )
                # If no initial_state is provided, dtype must be specified
                # outputs -> type list(tensor) shape: sequence_length, batch_size, hidden_size * 2

                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                # shape: batch_size, sequence_length, hidden_size * 2
                batch_size = tf.shape(outputs)[0]
                index = tf.range(0, batch_size) * \
                    sequence_length + (self.seq_len - 1)
                self.rnn_output = tf.gather(tf.reshape(
                    outputs, [-1, hidden_size * 2]), index)
                # shape: batch_size, hidden_size * 2
        else:
            if use_attention:
                with tf.name_scope("rnn-based-attention-with-{}-layers".format(num_layers)):
                    outputs, _, _ = tf.nn.static_bidirectional_rnn(
                        inputs=self.embedded_chars,
                        cell_fw=self.cell_fw,
                        cell_bw=self.cell_bw,
                        dtype=tf.float32
                    )

                    outputs = tf.stack(outputs)
                    outputs = tf.transpose(outputs, [1, 0, 2])
                    self.rnn_output, alpha = attention(outputs, attention_size)
            else:
                with tf.name_scope("rnn-with-{}-layers".format(num_layers)):
                    outputs, _, _ = tf.nn.static_bidirectional_rnn(
                        inputs=self.embedded_chars,
                        cell_fw=self.cell_fw,
                        cell_bw=self.cell_bw,
                        dtype=tf.float32
                    )
                    outputs = tf.stack(outputs)
                    outputs = tf.transpose(outputs, [1, 0, 2])
                    self.rnn_output = tf.reduce_sum(outputs, axis=1)

        with tf.name_scope("fully-connected-layer"):
            W = tf.Variable(tf.truncated_normal(
                [hidden_size * 2, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.rnn_output, W, b, name="scores")  # TODO error
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")