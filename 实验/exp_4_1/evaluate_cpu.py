import numpy as np
import struct
import os
import scipy.io
import time
import tensorflow as tf
from tensorflow.python.framework import graph_util

os.putenv('MLU_VISIBLE_DEVICES','')

IMAGE_PATH = 'data/cat1.jpg'
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5',

        'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
    )

    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        if name[:4] == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [height, width, in_channels, out_channels]
            # tensorflow: weights are [in_channels, height, width, out_channels]
            kernels = kernels.transpose(1, 0, 2, 3)
            current = _conv_layer(current, kernels, bias)
        elif name[:4] == 'relu':
            current = tf.nn.relu(current)
        elif name[:4] == 'pool':
            current = _pool_layer(current)
        elif name == 'softmax':
            current = tf.nn.softmax(current, axis=1)
        elif name  == 'fc6':
            kernels, bias = weights[i][0][0][0][0]
            current = tf.nn.bias_add(tf.matmul(tf.reshape(current, [-1, current.shape[1]*current.shape[2]*current.shape[3]]), tf.reshape(kernels, [kernels.shape[0]*kernels.shape[1]*kernels.shape[2], -1])), bias.flatten())
        elif name  == 'fc7':
            kernels, bias = weights[i][0][0][0][0]
            current = tf.nn.bias_add(tf.matmul(current, tf.reshape(kernels, [kernels.shape[0]*kernels.shape[1]*kernels.shape[2], -1])), bias.flatten())
        elif name  == 'fc8':
            kernels, bias = weights[i][0][0][0][0]
            current = tf.nn.bias_add(tf.matmul(current, tf.reshape(kernels, [kernels.shape[0]*kernels.shape[1]*kernels.shape[2], -1])), bias.flatten())

        net[name] = current 

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    return tf.nn.bias_add(tf.nn.conv2d(input, weights, 1, "SAME"), bias.flatten())

def _pool_layer(input):
    return tf.nn.max_pool(input, 2, 2, "VALID")

def preprocess(image,mean):
    return image - mean

def load_image(path):
    mean = np.array([123.68, 116.779, 103.939])
    image = scipy.misc.imresize(scipy.misc.imread(path), (224, 224, 3))
    image = preprocess(image, mean).reshape(-1, 224, 224, 3)
    return image

if __name__ == '__main__':
    input_image = load_image(IMAGE_PATH)

    with tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=(1,224,224,3),
                                         name='img_placeholder')
        nets = net(VGG_PATH, img_placeholder)

        for i in range(10):
            start = time.time()
            preds = sess.run(nets, feed_dict={img_placeholder:input_image})
            end = time.time()
            delta_time = end - start	
            print("processing time: %s" % delta_time)

        prob = preds['softmax'][0]
        top1 = np.argmax(prob)

        print('Classification result: id = %d, prob = %f' % (top1, prob[top1]))

        print("*** Start Saving Frozen Graph ***")
        # We retrieve the protobuf graph definition
        input_graph_def = sess.graph.as_graph_def()
        output_node_names = ["Softmax"]
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names,
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile("models/vgg19.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("**** Save Frozen Graph Done ****")