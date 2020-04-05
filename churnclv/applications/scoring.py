import pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from churnclv import BASE_PATH


def main():
    print('Loading Data')
    with open(BASE_PATH + '/output/datasets.pickle', 'rb') as handle:
        data = pickle.load(handle)

    model = tf.keras.models.load_model(BASE_PATH + '/trained_models/pointwise_model.h5')

    preds = model.predict(data['x_test_churn'].values)
    y_pred_lr = [x[0] for x in preds]
    y_true = data['y_test_churn'].values

    siamese = tf.keras.models.load_model(BASE_PATH + '/trained_models/siamese_model.h5', custom_objects={'tf': tf})
    print(siamese.summary())

    base_scorer = siamese.get_layer('model')

    preds = base_scorer.predict(data['x_test_churn'].values)
    y_pred_siam = [x[0] for x in preds]

    print('AUC on test data: ' + str(roc_auc_score(y_true, y_pred_lr)))
    print('AUC on test data: ' + str(roc_auc_score(y_true, y_pred_siam)))


if __name__ == '__main__':
    main()
