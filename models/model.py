import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.contrib.layers.python.layers import xavier_initializer as xavier


class convmodel_type():
    def __init__(self):
        growth = 16  # Growth Rate
        init = growth * 2  # Dense-BC
        growth4 = 4 * growth  # Dense-B
        blocksize = [6, 12, 24, 16]  # Dense-121

        with tf.variable_scope("model1"):
            self.w1 = tf.get_variable(name='w1', shape=[3, 3, 3, 1, init], initializer=xavier())

            self.w2 = []
            for i in range(0, blocksize[0]):
                self.w2.append(tf.get_variable(name='w2_1x1_' + str(i), shape=[1, 1, 1, init + i * growth, growth4],
                                               initializer=xavier()))
                self.w2.append(
                    tf.get_variable(name='w2_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init1 = (init + blocksize[0] * growth) / 2
            self.w3_1x1 = tf.get_variable(name='w3_1x1', shape=[1, 1, 1, init + blocksize[0] * growth, init1],
                                          initializer=xavier())  # Dense-C

            self.w3 = []
            for i in range(0, blocksize[1]):
                self.w3.append(tf.get_variable(name='w3_1x1_' + str(i), shape=[1, 1, 1, init1 + i * growth, growth4],
                                               initializer=xavier()))
                self.w3.append(
                    tf.get_variable(name='w3_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init2 = (init1 + blocksize[1] * growth) / 2
            self.w4_1x1 = tf.get_variable(name='w4_1x1', shape=[1, 1, 1, init1 + blocksize[1] * growth, init2],
                                          initializer=xavier())

            self.w4 = []
            for i in range(0, blocksize[2]):
                self.w4.append(tf.get_variable(name='w4_1x1_' + str(i), shape=[1, 1, 1, init2 + i * growth, growth4],
                                               initializer=xavier()))
                self.w4.append(
                    tf.get_variable(name='w4_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init3 = (init2 + blocksize[2] * growth) / 2
            self.w5_1x1 = tf.get_variable(name='w5_1x1', shape=[1, 1, 1, init2 + blocksize[2] * growth, init3],
                                          initializer=xavier())

            self.w5 = []
            for i in range(0, blocksize[3]):
                self.w5.append(tf.get_variable(name='w5_1x1_' + str(i), shape=[1, 1, 1, init3 + i * growth, growth4],
                                               initializer=xavier()))
                self.w5.append(
                    tf.get_variable(name='w5_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            self.fc1 = tf.get_variable(name='fc1', shape=[1, 1, 1, init3 + blocksize[3] * growth, 3],
                                       initializer=xavier())
            self.fc1b = tf.get_variable(name='fc1b', shape=[3], initializer=xavier())

    def denseblock(self, input, kernel1, kernel2, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel1, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')

            c = self.SE_block(c,c._shape_as_list()[-1],16,layer_name+'SE')

            input = tf.concat([input, c], axis=4)

        return input

    def add_transition(self, c, kernel, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name)
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = tf.nn.avg_pool3d(c, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')

        return c

    def convnet(self):

        image = tf.placeholder("float32", [None, 32, 48, 48, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        # expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')

        for i in range(0, len(self.w2), 2):
            conv = self.denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)

        conv = self.add_transition(conv, self.w3_1x1, 'conv3_1x1', is_training)

        for i in range(0, len(self.w3), 2):
            conv = self.denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)

        conv = self.add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)

        for i in range(0, len(self.w4), 2):
            conv = self.denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)

        conv = self.add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        for i in range(0, len(self.w5), 2):
            conv = self.denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)

        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)

        return image, is_training, tf.nn.softmax(conv, dim=-1)

    def Fully_connected(self, x, units, layer_name):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=False, units=units)

    def SE_block(self, input_x, out_dim, ratio, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):

            in_shape = input_x._shape_as_list()
            squeeze = tf.nn.avg_pool3d(input_x,[1, in_shape[1], in_shape[2], in_shape[3], 1],\
                                       strides=[1 ,1 ,1 ,1 ,1], padding='VALID')
            excitation = self.Fully_connected(squeeze, out_dim // ratio, layer_name + '_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = self.Fully_connected(excitation, out_dim, layer_name + '_fully_connected2')
            excitation = tf.nn.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1, 1, 1, 1, out_dim])
            # print 'reshape shape', excitation._shape_as_list()

            scale = input_x * excitation

        return scale



class convmodel():
    def __init__(self):
        growth = 16  # Growth Rate
        init = growth * 2  # Dense-BC
        growth4 = 4 * growth  # Dense-B
        blocksize = [6, 12, 24, 16]  # Dense-121

        with tf.variable_scope("model1"):
            self.w1 = tf.get_variable(name='w1', shape=[3, 3, 3, 1, init], initializer=xavier())

            self.w2 = []
            for i in range(0, blocksize[0]):
                self.w2.append(tf.get_variable(name='w2_1x1_' + str(i), shape=[1, 1, 1, init + i * growth, growth4],
                                               initializer=xavier()))
                self.w2.append(
                    tf.get_variable(name='w2_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init1 = (init + blocksize[0] * growth) / 2
            self.w3_1x1 = tf.get_variable(name='w3_1x1', shape=[1, 1, 1, init + blocksize[0] * growth, init1],
                                          initializer=xavier())  # Dense-C

            self.w3 = []
            for i in range(0, blocksize[1]):
                self.w3.append(tf.get_variable(name='w3_1x1_' + str(i), shape=[1, 1, 1, init1 + i * growth, growth4],
                                               initializer=xavier()))
                self.w3.append(
                    tf.get_variable(name='w3_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init2 = (init1 + blocksize[1] * growth) / 2
            self.w4_1x1 = tf.get_variable(name='w4_1x1', shape=[1, 1, 1, init1 + blocksize[1] * growth, init2],
                                          initializer=xavier())

            self.w4 = []
            for i in range(0, blocksize[2]):
                self.w4.append(tf.get_variable(name='w4_1x1_' + str(i), shape=[1, 1, 1, init2 + i * growth, growth4],
                                               initializer=xavier()))
                self.w4.append(
                    tf.get_variable(name='w4_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            init3 = (init2 + blocksize[2] * growth) / 2
            self.w5_1x1 = tf.get_variable(name='w5_1x1', shape=[1, 1, 1, init2 + blocksize[2] * growth, init3],
                                          initializer=xavier())

            self.w5 = []
            for i in range(0, blocksize[3]):
                self.w5.append(tf.get_variable(name='w5_1x1_' + str(i), shape=[1, 1, 1, init3 + i * growth, growth4],
                                               initializer=xavier()))
                self.w5.append(
                    tf.get_variable(name='w5_' + str(i), shape=[3, 3, 3, growth4, growth], initializer=xavier()))

            self.fc1 = tf.get_variable(name='fc1', shape=[1, 1, 1, init3 + blocksize[3] * growth, 2],
                                       initializer=xavier())
            self.fc1b = tf.get_variable(name='fc1b', shape=[2], initializer=xavier())

    def _denseblock(self, input, kernel1, kernel2, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(input, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '1')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel1, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name + '2')
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel2, strides=(1, 1, 1, 1, 1), padding='SAME')
            input = tf.concat([input, c], axis=4)

        return input

    def _add_transition(self, c, kernel, layer_name, is_training):

        with tf.name_scope(layer_name):
            c = batch_norm(c, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                           is_training=is_training, scope=layer_name)
            c = tf.nn.relu(c)
            c = tf.nn.conv3d(c, kernel, strides=(1, 1, 1, 1, 1), padding='SAME')
            c = tf.nn.avg_pool3d(c, (1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1), padding='SAME')

        return c

    def _convnet(self):

        image = tf.placeholder("float32", [None, 32, 48, 48, 1])
        id = tf.placeholder("int32", [None, 1])
        is_training = tf.placeholder(tf.bool, shape=())
        expand_dims = tf.placeholder(tf.bool, shape=())

        conv = tf.nn.conv3d(image, self.w1, strides=(1, 1, 1, 1, 1), padding='SAME')

        for i in range(0, len(self.w2), 2):
            conv = self._denseblock(conv, self.w2[i], self.w2[i + 1], 'conv2_' + str(i), is_training)

        conv = self._add_transition(conv, self.w3_1x1, 'conv3_1x1', is_training)

        for i in range(0, len(self.w3), 2):
            conv = self._denseblock(conv, self.w3[i], self.w3[i + 1], 'conv3_' + str(i), is_training)

        conv = self._add_transition(conv, self.w4_1x1, 'conv4_1x1', is_training)

        for i in range(0, len(self.w4), 2):
            conv = self._denseblock(conv, self.w4[i], self.w4[i + 1], 'conv4_' + str(i), is_training)

        conv = self._add_transition(conv, self.w5_1x1, 'conv5_1x1', is_training)

        for i in range(0, len(self.w5), 2):
            conv = self._denseblock(conv, self.w5[i], self.w5[i + 1], 'conv5_' + str(i), is_training)

        conv = batch_norm(conv, data_format='NHWC', center=True, scale=True, updates_collections='_update_ops_',
                          is_training=is_training, scope='bn_last')
        conv = tf.nn.relu(conv)
        shape = conv.get_shape().as_list()
        conv = tf.nn.avg_pool3d(conv, (1, shape[1], shape[2], shape[3], 1), strides=(1, 1, 1, 1, 1), padding='VALID')

        conv = tf.nn.conv3d(conv, self.fc1, strides=(1, 1, 1, 1, 1), padding='VALID') + self.fc1b
        conv = tf.squeeze(conv)

        return image, is_training, tf.nn.softmax(conv, dim=-1)
