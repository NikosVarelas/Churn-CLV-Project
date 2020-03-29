import tensorflow as tf


class PairwiseModel(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def create_base_network(self):
        input_features = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activation='sigmoid',
            name='score')(input_features)
        return tf.keras.models.Model(input_features, x)

    def siamese_network(self):
        positive_features = tf.keras.Input(shape=self.input_shape, name='positive_customer')
        negative_features = tf.keras.Input(shape=self.input_shape, name='negative_customer')

        base_scorer = self.create_base_network()
        positive_score = base_scorer(positive_features)
        negative_score = base_scorer(negative_features)

        score_diff = tf.keras.layers.Lambda(
            lambda x: tf.math.sigmoid(x[0]-x[1]),
            name='score_diff'
        )([positive_score, negative_score])

        return tf.keras.models.Model(inputs=[positive_features, negative_features],
                                     outputs=score_diff)


def create_pairs(x, y):
    positive_customer, negative_customer= tf.dynamic_partition(x, y, 2)
    len_pos = len(positive_customer)
    len_neg = len(negative_customer)

    positive_customer = [x for i in range(len_neg) for x in positive_customer]
    label = tf.ones([len(positive_customer), 1])
    negative_customer = [x for i in range(len_pos) for x in negative_customer]

    return positive_customer, negative_customer, label
