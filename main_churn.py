from utils import *

if __name__ == "__main__":
    print("Reading Data")
    data = pd.read_csv('./data/customerData.csv', sep=',')
    data['Transaction_date'] = pd.to_datetime(data['Transaction_date'])

    # Predicting CLV one month ahead

    print("Started preproc")
    preprocess_train = Preprocessor(data, 1, 'Transaction_date', 'Basket_id', 14)
    preprocess_labels = LabelEstimator(data, 1, 'Transaction_date', 'Basket_id', 14)
    train_set, predict_set = preprocess_train.fit('Customer_no', 'item_net_amount')
    labels = preprocess_labels.fit('Customer_no', 'item_net_amount')

    # Dropping customers with less than 5 events
    print(train_set.head())
    print(predict_set.head())
    print(labels.head())