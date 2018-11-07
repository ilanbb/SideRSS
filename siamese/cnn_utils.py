import numpy as np
import tensorflow as tf


def create_conv_layer(data, weights_matrix, bias_vector, strides_x_y, is_max=False):
    '''
    Helper function to construct a CNN layer (convolution and possible max pooling)

    Args:
        data: data tensor
        weights_matrix: convolution weights/filter (as tensor)
        bias_vector: convolution bias (as tensor)
        strides_x_y: convlution strides (tuple with x-value and y-value strides)
        is_max: whether to perform max pooling on the second axis (get the max value on all positions)
       
    Returns:
        results: convolution output (as tensor of shape [batch_size, max_seq_len - filter_x + 1, 1, output_channels] (if no max pooling))
    '''
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.conv2d(data, weights_matrix, strides=all_strides, padding='VALID')
    result = tf.nn.bias_add(result, bias_vector)
    result = tf.nn.relu(result)
    if is_max:
        result = tf.expand_dims(tf.reduce_max(result, axis=1), axis=1)
    return result


def nn_layer(data, weights, bias, activate_relu):
    '''
    Helper function to perform a linear transformation with some non linear activation

    Args:
        data: data tensor 
        weights: weights tensor
        bias: bias tensor
        activate_relu: whether to activate ReLU or Sigmoid non-linear function 
    Return:
        result: result of data transformation
    '''
    result = tf.add(tf.matmul(data, weights), bias)
    if activate_relu:
        result = tf.nn.relu(result)
    else:
        result = tf.nn.sigmoid(result)
    return result
                           
