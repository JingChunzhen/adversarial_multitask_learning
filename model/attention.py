import tensorflow as tf


def attention(inputs, attention_size):
    '''
    cf. http://www.aclweb.org/anthology/N16-1174
    Args:
        inputs (tensor dtype=tf.float32): shape (batch_size, sequence_length, (2 *) hidden_size)
    Returns:
        the attention output tensor: shape (batch_size, (2 *) hidden_size)
        with attention weight: shape (batch_size, sequence_length)
    '''
    hidden_size = inputs.shape[2].value
    w1 = tf.Variable(tf.random_normal(
        [hidden_size, attention_size], stddev=0.1), dtype=tf.float32)
    w2 = tf.Variable(tf.random_normal(
        [attention_size], stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.random_normal(
        [attention_size], stddev=0.1), dtype=tf.float32)

    output = tf.tensordot(inputs, w1, [[2], [0]])
    # shape: batch_size, sequence_length, attention_size
    output = tf.nn.tanh(output + b1)
    output = tf.tensordot(output, w2, [[2], [0]])
    # shape: batch_size, sequence_length
    alphas = tf.nn.softmax(output)
    # shape: batch_size, sequence_length

    output = tf.multiply(inputs, tf.expand_dims(alphas, -1))
    # shape: batch_size, sequence_length, hidden_size
    output = tf.reduce_sum(output, axis=1)
    # shape: batch_size, hidden_size
    return output, alphas

