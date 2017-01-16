#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse
import sys
FLAGS = None
learning_rate = 0.001
log_dir = './'
max_steps = 5000


def weight_initialize(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_initialize(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):

    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weight = weight_initialize([input_dim,output_dim])
            variable_summaries(weight)
        with tf.name_scope("bias"):
            biases = bias_initialize([output_dim])
            variable_summaries(biases)
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weight) + biases
            tf.summary.histogram("pre_activations", preactivate)
        activations = act(preactivate,name="activation")
        tf.summary.histogram("activations",activations)

        return activations


def  Neural(label_list, state_dict, start_index, data, n_features):
    categories = len(state_dict.keys())
    print 'len(categories)',categories
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None,n_features], name="x_input")
        y_ = tf.placeholder(tf.float32, [None,categories], name="y_input")

    hidden1_layer = nn_layer(x, n_features, 20, "layer1")
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_prob",keep_prob)
        dropped = tf.nn.dropout(hidden1_layer, keep_prob)
    y = nn_layer(dropped, 20, categories, "layer2", act=tf.identity)
    with tf.name_scope("cross_entropy"):
        diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
    	with tf.name_scope('correct_prediction'):
      	    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    tf.initialize_all_variables().run()
    
    for i in range(max_steps):
        if i % 99  == 0:
            train_accuracy = accuracy.eval(feed_dict = \
                {x:data[0:start_index,:], y_:label_list, keep_prob:1.0})
            print("step %d, training accuary %g",(i, train_accuracy))
            summary_str = sess.run(merged, feed_dict=\
                {x:data[0:start_index,:],y_:label_list,keep_prob:1.0})
            train_writer.add_summary(summary_str, i)
            train_writer.flush()

        train.run(feed_dict={x:data[0:start_index,:], y_:label_list,keep_prob:0.8})
    train_writer.close()
    #run test
    predict_label = sess.run(tf.argmax(tf.nn.softmax(y), 1),feed_dict={x:data[start_index:],keep_prob:1.0})
    print predict_label
    return predict_label


def main(_):
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    Neural()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')

  parser.add_argument('--log_dir', type=str, default='./nn_hmm_log',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




