

A very good post [here](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)

In short, just use dynamic_rnn

## Bug Report

### No variable to optimize

this is caused by tf.variable_scope(), when using tf.nn.static_rnn

### Attempting to use uninitialized value beta2_power2

this is caused by using AdamOptimizer

### slow loading speed of the graph

use dynamic_rnn instead 