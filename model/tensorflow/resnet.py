import tensorflow as tf

import numpy as np
from functools import reduce
from tensorflow.contrib.layers.python.layers import batch_norm



class ResNet:
    """
    A trainable version VGG19.
    """

    def __init__(self, resnet_npy_path=None, trainable=True, dropout=0.5):
        if resnet_npy_path is not None:
            self.data_dict = np.load(resnet_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build_tower(self, images, train_mode = None):
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [3, 3, 3, 3, 3]
        strides = [1, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self.conv_layer(images, kernels[0],[1,strides[0],strides[0],1],3,filters[0],'conv0')
            x = self.batch_norm_layer(x, train_mode, 'bn0')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x,train_mode,filters[0], name='conv2_1')
        x = self._residual_block(x,train_mode,filters[0], name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, train_mode,filters[0], filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x,train_mode, filters[2], name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, train_mode,filters[2], filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, train_mode, filters[3], name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x,train_mode, filters[3], filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x,train_mode,filters[4], name='conv5_2')

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            x = self.fc_layer(x, filters[4], 1000,'fc')

        logits = x

        return logits


    def _residual_block_first(self, x,train_mode, in_channel, out_channel, strides, name="unit"):
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self.conv_layer(x, 1,[1, strides, strides, 1],in_channel,out_channel,'shortcut')
            # Residual
            x = self.conv_layer(x, 3,[1, strides, strides, 1],in_channel,out_channel,'conv_1')
            x = self.batch_norm_layer(x, train_mode, 'bn_1')
            x = tf.nn.relu(x)
            x = self.conv_layer(x, 3,[1,1,1,1],out_channel,out_channel,'conv_2')
            x = self.batch_norm_layer(x, train_mode, 'bn_2')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x)
        return x


    def _residual_block(self, x,train_mode, in_channel, input_q=None, output_q=None, name="unit"):
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self.conv_layer(x, 3,[1,1,1,1],in_channel,in_channel,'conv_1')
            x = self.batch_norm_layer(x, train_mode,'bn_1')
            x = tf.nn.relu(x)
            x = self.conv_layer(x, 3,[1,1,1,1],in_channel,in_channel,'conv_2')
            x = self.batch_norm_layer(x, train_mode,'bn_2')

            x = x + shortcut
            x = tf.nn.relu(x)
        return x


    def batch_norm_layer(self, x, train_phase, scope_bn):
        return batch_norm(x, decay=0.9, center=True, scale=True,
                    updates_collections=None,
                    is_training=train_phase,
                    reuse=None,
                    trainable=True,
                    scope=scope_bn)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, filter_size, stride, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, np.sqrt(2./(filter_size*filter_size*out_channels)))
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 1.0/out_size)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
