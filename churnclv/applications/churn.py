from pipeline import Pipeline
import pandas as pd

if __name__ == '__main__':
    print("Reading Data")
    data = pd.read_csv('./data/customerData.csv', sep=',')
    data['Transaction_date'] = pd.to_datetime(data['Transaction_date'])
    pipeline = Pipeline(data=data,
                        months=1,
                        date_col='Transaction_date',
                        basket_col='Basket_id',
                        churn_days=14,
                        lag=1
                        )



