import tensorflow as tf


class PairwiseModel(object):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    @staticmethod
    def create_pairs(x, y):
        positive, negative = tf.dynamic_partition(x, y, 2)
        return positive, negative

    def create_base_network(self):
        input_features = tf.keras.Input(shape=self.input_shape)
        x_hidden = tf.keras.layers.Dense(
            4,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activation='sigmoid')(input_features)

        x = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            activation='sigmoid')(x_hidden)
        return tf.keras.models.Model(input_features, x)

