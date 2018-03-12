import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]

input_x = tf.placeholder(tf.float32, [None, 10, 8])

seq_len = tf.placeholder(tf.int32, [None])

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

outputs, states  = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell,
    cell_bw=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

output_fw, output_bw = outputs
states_fw, states_bw = states

print(tf.shape(output_fw))

# result = tf.contrib.learn.run_n(
#     {"output_fw": output_fw, "output_bw": output_bw, "states_fw": states_fw, "states_bw": states_bw},
#     n=1,
#     feed_dict=None)

final_output = tf.concat([output_fw, output_bw], axis=2)
print(tf.shape(output_bw))
print(tf.shape(final_output))

batch_size = tf.shape(outputs)[0]
index = tf.range(0, batch_size) * \
    10 + (seq_len - 1)

output = tf.gather(tf.reshape(
    final_output, [-1, 64 * 2]), index)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  
    output_fw_, fin, out, ind = sess.run(
        [output_fw, final_output, output, index], 
        feed_dict={
            input_x: X,
            seq_len: X_lengths
        }
    )

    print(np.shape(output_fw_))
    print(np.shape(fin))
    print(np.shape(out))

print(ind)
print(output_fw_)
print(out)


"""
(2, 10, 64)
(2, 10, 64)
(2, 64)
(2, 64)
"""

