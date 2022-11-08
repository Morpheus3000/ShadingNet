import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops	
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


def image_gradients(image):
        if image.get_shape().ndims != 4:
                raise ValueError('image_gradients expects a 4D tensor [batch_size, h, w, d], not %s.', image.get_shape())
        image_shape = array_ops.shape(image)
        batch_size, height, width, depth = array_ops.unstack(image_shape)
        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]

        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
        dx = array_ops.reshape(dx, image_shape)

        return dy, dx


def efficient_channel_attention(features, W, gamma=2, b=1):

    gap = tf.reduce_mean(features, [1, 2]) # global average pooling
    gap = tf.reshape(gap, [-1,1,features.get_shape()[3]])

    conv = tf.nn.conv1d(gap, W, stride=1, padding='SAME')

    scale = tf.nn.sigmoid(conv)
    scale = tf.reshape(scale, [-1,1,1,features.get_shape()[3]])

    attention = features * scale

    return tf.nn.relu(attention)


def channel_wise_attention(composite, prior, W1, W2, isTraining):
	units = prior.get_shape()[3]/2

	conv1 = tf.nn.conv2d(prior, W1, strides=[1, 1, 1, 1], padding='SAME')
	conv2 = tf.nn.conv2d(prior, W2, strides=[1, 1, 1, 1], padding='SAME')

	gap = tf.reduce_mean(composite, [1, 2]) # global average pooling

	fc = tf.layers.dense(inputs=gap, use_bias=False, units=units)
	fc = tf.reshape(fc, [-1,1,1,units])

	scale1 = conv1 * tf.nn.sigmoid(fc)

	scale2 = conv2 * scale1

	bn = tf.contrib.layers.batch_norm(scale2, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
	relu = tf.nn.relu(bn)

	return relu


def create_bottleneck(rgb_features, prior_features, W1, W2, isTraining):
	fused = tf.concat([rgb_features, prior_features], 3)
	fusion = tf.nn.conv2d(fused, W2, strides=[1, 1, 1, 1], padding='SAME')

	rgb_stuff = tf.nn.conv2d(rgb_features, W1, strides=[1, 1, 1, 1], padding='SAME')

	AT = contextual_attention(rgb_stuff, prior_features, training=isTraining)

	return tf.concat([fusion, AT], 3)


def dilated_res(x, W1, W2, d, isTraining):
	bn1 = tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
	relu1 = tf.nn.relu(bn1)
	conv1 = tf.nn.atrous_conv2d(relu1, W1, d, padding='SAME')
	bn2 = tf.contrib.layers.batch_norm(conv1, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
	relu2 = tf.nn.relu(bn2)
	conv2 = tf.nn.atrous_conv2d(relu2, W2, d, padding='SAME')

	return tf.add(x, conv2)


def res(x, W1, W2, isTraining):
	bn1 = tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
	relu1 = tf.nn.relu(bn1)
	conv1 = tf.nn.conv2d(relu1, W1, strides=[1, 1, 1, 1], padding='SAME')
	bn2 = tf.contrib.layers.batch_norm(conv1, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
	relu2 = tf.nn.relu(bn2)
	conv2 = tf.nn.conv2d(relu2, W2, strides=[1, 1, 1, 1], padding='SAME')

	return tf.add(x, conv2)
	

def conv2d(x, W, isTraining, s=1):
	conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
	bn = tf.contrib.layers.batch_norm(conv, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
	relu = tf.nn.leaky_relu(bn, 0.01)

	return relu


def conv2d_tanh(x, W, isTraining, s=1):
    conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
    bn = tf.contrib.layers.batch_norm(conv, decay=0.9, scale=True, is_training=isTraining, updates_collections=None)
    tanh = tf.nn.tanh(bn)

    return tanh


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.Tensor: output
    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    #f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    #b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    #if mask is not None:
    #    mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask
	yi = tf.where(tf.is_nan(yi), tf.zeros_like(yi), yi)
	yi = tf.where(tf.is_inf(yi), tf.zeros_like(yi), yi)

        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    y = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
    y = tf.where(tf.is_inf(y), tf.zeros_like(y), y)


    return y
