from pipeline import Pipeline
import pandas as pd
from churnclv.utils.feature_engineering import train_valid_test_split


def main():
    print("Reading Data")
    data = pd.read_csv('./data/customerData.csv', sep=',')
    data['Transaction_date'] = pd.to_datetime(data['Transaction_date'])
    pipeline = Pipeline(data=data,
                        months=1,
                        date_col='Transaction_date',
                        basket_col='Basket_id',
                        churn_days=14,
                        lag=1)
    pipeline.fit('Customer_no', 'item_net_amount')
    train, predict = pipeline.transform('Customer_no', 'item_net_amount')
    x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(
        train, 'churn', 0.3, stratify_fold=True)


if __name__ == '__main__':
    main()
