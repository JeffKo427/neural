# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf
import numpy as np

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    return tf.Variable( tf.truncated_normal( (n_features, n_labels) ) )


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    return tf.Variable( tf_zeros(n_labels) )


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    return tf.add( tf.matmul( input, w ), b )

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    return np.exp(x) / np.sum( np.exp(x), axis = 0 )

logits = [3.0, 1.0, 0.2]
print(softmax(logits))