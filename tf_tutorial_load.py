import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('tf_tutorial_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
print(sess.graph.get_operations())


x, y_, keep_prob = tf.get_collection('placeholders')
accuracy = tf.get_collection('accuracy')[0]

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
