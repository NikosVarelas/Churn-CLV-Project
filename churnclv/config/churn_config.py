import os
os.getcwd()


class ChurnConfig(object):
    def __init__(self, transactions_path):
        self.transactions_path = transactions_path


churn_config = ChurnConfig(
    transactions_path='')


if __name__ == '__main__':
    print(os.getcwd())