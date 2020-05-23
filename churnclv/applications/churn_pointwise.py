import pickle
import os
import tensorflow as tf

from churnclv import BASE_PATH
from churnclv.models.pairwise import PairwiseModel


def main():
    print('Loading Data')
    with open(BASE_PATH + '/output/datasets.pickle', 'rb') as handle:
        data = pickle.load(handle)

    pairwise = PairwiseModel(input_shape=8)
    model = pairwise.create_base_network()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy',
                           tf.keras.metrics.AUC(name='auc')])

    model.fit(data['x_train_churn'].values,
              data['y_train_churn'].values,
              validation_data=(data['x_val_churn'].values,
                               data['y_val_churn'].values),
              epochs=500)

    if not os.path.isdir(BASE_PATH + '/trained_models'):
        os.mkdir(BASE_PATH + '/trained_models')
    model.save(BASE_PATH + '/trained_models/' + 'pointwise_model.h5')


if __name__ == '__main__':
    main()