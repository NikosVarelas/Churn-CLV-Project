import pickle
import os
import tensorflow as tf

from churnclv import BASE_PATH
from churnclv.models.pairwise import PairwiseModel
from churnclv.models.pairwise import create_pairs


def main():
    print('Loading Data')
    with open(BASE_PATH + '/output/datasets.pickle', 'rb') as handle:
        data = pickle.load(handle)

    pairwise = PairwiseModel(input_shape=8)

    # Create the training set
    positive_train, negative_train, label_train = create_pairs(
        data['x_train_churn'].values, data['y_train_churn'].values)

    # Create the validation set
    positive_val, negative_val, label_val = create_pairs(
        data['x_val_churn'].values, data['y_val_churn'].values)

    siamese = pairwise.siamese_network()
    siamese.compile(loss='binary_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

    print(siamese.summary())

    def scheduler(epoch):
        if epoch < 50:
            return 0.1
        else:
            return float(0.1 * tf.math.exp(0.1 * (10 - epoch)))

    wd = tf.keras.callbacks.LearningRateScheduler(scheduler)

    siamese.fit(
        [positive_train, negative_train],
        label_train,
        validation_data=([positive_val, negative_val], label_val),
        callbacks=[wd],
        epochs=100)

    if not os.path.isdir(BASE_PATH + '/trained_models'):
        os.mkdir(BASE_PATH + '/trained_models')
    siamese.save(BASE_PATH + '/trained_models/' + 'siamese_model.h5')


if __name__ == '__main__':
    main()
