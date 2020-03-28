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
        x = tf.keras.layers.Dense(1, activation='sigmoid')(input_features)
        return tf.keras.models.Model(input_features, x)

