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
        x = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activation='sigmoid')(input_features)
        return tf.keras.models.Model(input_features, x)

    def siamese_network(self):
        positive_features = tf.keras.Input(shape=self.input_shape, name='positive_customer')
        negative_features = tf.keras.Input(shape=self.input_shape, name='negative_customer')

        base_scorer = self.create_base_network()
        positive_score = base_scorer(positive_features)
        negative_score = base_scorer(negative_features)

        score_diff = tf.keras.layers.Lambda(
            lambda x: x[0]-x[1],
            name='score_diff'
        )([positive_score, negative_score])

        return tf.keras.models.Model(input=[positive_features, negative_features],
                                     output=score_diff)
