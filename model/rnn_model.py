import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from model.attention import attention


class RNN(object):

    def __init__(self, sequence_length, 
                 hidden_size, num_layers, input_keep_prob=1, output_keep_prob=1, dynamic, use_attention,
                 attention_size):
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dynamic = dynamic
        self.use_attention = use_attention
        self.attention_size = attention_size
                
        #l2_loss = tf.constant(0.0)
                        
        with tf.name_scope("forward-cell"):
            if num_layers != 1:
                cells = []
                for i in range(num_layers):
                    rnn_cell = DropoutWrapper(
                        GRUCell(hidden_size),
                        input_keep_prob=input_keep_prob,
                        output_keep_prob=output_keep_prob
                    )
                    cells.append(rnn_cell)
                self.cell_fw = MultiRNNCell(cells)
            else:
                self.cell_fw = DropoutWrapper(
                    GRUCell(hidden_size),
                    input_keep_prob=input_keep_prob,
                    output_keep_prob=output_keep_prob
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

    def process(self, x, seq_len):
        """
        Args:
            x (tensor of list): shape (batch_size, sequence_length, embedding_size)
            seq_len (tensor of list): shape (batch_size, 1)
        """
        x = tf.unstack(x, self.sequence_length, axis=1)
        # get list (length == sequence_length) of tensors with shape: batch_size, embedding_size
        if self.dynamic:
            with tf.name_scope("dynamic-rnn-with-{}-layers".format(self.num_layers)):
                outputs, _, _ = tf.nn.static_bidirectional_rnn(
                    inputs=x,
                    cell_fw=self.cell_fw,
                    cell_bw=self.cell_bw,
                    sequence_length=seq_len,
                    dtype=tf.float32
                )
                # If no initial_state is provided, dtype must be specified
                # outputs -> type list(tensor) shape: sequence_length, batch_size, hidden_size * 2

                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                # shape: batch_size, sequence_length, hidden_size * 2
                batch_size = tf.shape(outputs)[0]
                index = tf.range(0, batch_size) * \
                    self.sequence_length + (seq_len - 1)
                output = tf.gather(tf.reshape(
                    outputs, [-1, self.hidden_size * 2]), index)
                # shape: batch_size, hidden_size * 2
        else:
            if self.use_attention:
                with tf.name_scope("rnn-based-attention-with-{}-layers".format(self.num_layers)):
                    outputs, _, _ = tf.nn.static_bidirectional_rnn(
                        inputs=x,
                        cell_fw=self.cell_fw,
                        cell_bw=self.cell_bw,
                        dtype=tf.float32
                    )

                    outputs = tf.stack(outputs)
                    outputs = tf.transpose(outputs, [1, 0, 2])
                    output, alpha = attention(outputs, self.attention_size)
            else:
                with tf.name_scope("rnn-with-{}-layers".format(self.num_layers)):
                    outputs, _, _ = tf.nn.static_bidirectional_rnn(
                        inputs=x,
                        cell_fw=self.cell_fw,
                        cell_bw=self.cell_bw,
                        dtype=tf.float32
                    )
                    outputs = tf.stack(outputs)
                    outputs = tf.transpose(outputs, [1, 0, 2])
                    output = tf.reduce_sum(outputs, axis=1)
        return output
        