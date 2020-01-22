from churnclv.applications.pipeline import Pipeline
import pandas as pd


def main():
    print("Reading Data")
    data = pd.read_csv('../../data/customerData.csv', sep=',')
    data['Transaction_date'] = pd.to_datetime(data['Transaction_date'])
    print(data.head())
    pipeline = Pipeline(data=data,
                        months=1,
                        date_col='Transaction_date',
                        basket_col='Basket_id',
                        churn_days=14,
                        lag=1,
                        n_components=8)

    pipeline.fit('Customer_no', 'item_net_amount')
    x_train, x_val, x_test, x_test, x_pred, y_train, y_val, y_test = pipeline.transform('Customer_no', 'item_net_value', 'churn')
    print(x_train.head())


if __name__ == '__main__':
    main()



