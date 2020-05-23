from churnclv.applications.pipeline import Pipeline
import pandas as pd
import pickle
import os
from churnclv import BASE_PATH


def main():
    print("Reading Data")
    data = pd.read_csv(BASE_PATH+'/data/customerData.csv', sep=',')
    data['Transaction_date'] = pd.to_datetime(data['Transaction_date'])
    pipeline = Pipeline(data=data,
                        months=1,
                        date_col='Transaction_date',
                        basket_col='Basket_id',
                        churn_days=14,
                        lag=1,
                        n_components=8)

    pipeline.fit('Customer_no', 'item_net_amount')
    train_set, predict_set = pipeline.create_sets('Customer_no')
    x_train_chrun, x_val_churn, x_test_churn, x_pred_churn, y_train_churn, y_val_churn, y_test_chrun = pipeline.transform(
        key='Customer_no',
        train_set=train_set,
        predict_set=predict_set,
        target='churn')

    x_train_clv, x_val_clv, x_test_clv, x_pred_clv, y_train_clv, y_val_clv, y_test_clv = pipeline.transform(
        key='Customer_no',
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
    main()
