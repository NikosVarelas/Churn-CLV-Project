from churnclv.applications.pipeline import Pipeline
import pandas as pd
import pickle
import os

from churnclv.config.churn_config import config
from churnclv import BASE_PATH


def main(configuration):
    print("Reading Data")
    data = pd.read_csv(configuration.data_path, sep=',')
    data[configuration.transaction_date] = pd.to_datetime(data[configuration.transaction_date])
    pipeline = Pipeline(data=data,
                        months=configuration.months_to_predict,
                        date_col=configuration.transaction_date,
                        basket_col=configuration.basket_id,
                        churn_days=configuration.define_churn_days,
                        lag=1,
                        n_components=configuration.pca_components)

    pipeline.fit(configuration.customer_id, configuration.item_amount)
    train_set, predict_set = pipeline.create_sets(configuration.customer_id)
    x_train_chrun, x_val_churn, x_test_churn, x_pred_churn, y_train_churn, y_val_churn, y_test_chrun = pipeline.transform(
        key=configuration.customer_id,
        train_set=train_set,
        predict_set=predict_set,
        target='churn')

    x_train_clv, x_val_clv, x_test_clv, x_pred_clv, y_train_clv, y_val_clv, y_test_clv = pipeline.transform(
        key=configuration.customer_id,
        train_set=train_set,
        predict_set=predict_set,
        target='clv')

    data_dict = {
        'x_train_churn': x_train_chrun,
        'x_val_churn': x_val_churn,
        'x_test_churn': x_test_churn,
        'x_pred_churn': x_pred_churn,
        'y_train_churn': y_train_churn,
        'y_val_churn': y_val_churn,
        'y_test_churn': y_test_chrun,
        'x_train_clv': x_train_clv,
        'x_val_clv': x_val_clv,
        'x_test_clv': x_test_clv,
        'x_pred_clv': x_pred_clv,
        'y_train_clv': y_train_clv,
        'y_val_clv': y_val_clv,
        'y_test_clv': y_test_clv
    }

    if not os.path.isdir(BASE_PATH + '/output'):
        os.mkdir(BASE_PATH + '/output')

    with open(BASE_PATH + '/output/datasets.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(config)
