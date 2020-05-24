import yaml

from churnclv import BASE_PATH
import os
os.getcwd()


class ChurnConfig(object):
    def __init__(self, data_path, transaction_date, basket_id, customer_id, item_amount,
                 months_to_predict, define_churn_days, pca_components):
        self.data_path = data_path
        self.transaction_date = transaction_date
        self.basket_id = basket_id
        self.customer_id = customer_id
        self.item_amount = item_amount
        self.months_to_predict = months_to_predict
        self.define_churn_days = define_churn_days
        self.pca_components = pca_components


with open(BASE_PATH + '/resources/configs/churn_clv.yml', 'r') as ymlfile:
    yaml_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

config = ChurnConfig(data_path=BASE_PATH + yaml_config['preproc']['data_path'],
                     transaction_date=yaml_config['preproc']['transaction_date'],
                     basket_id=yaml_config['preproc']['basket_id'],
                     customer_id=yaml_config['preproc']['customer_id'],
                     item_amount=yaml_config['preproc']['item_amount'],
                     months_to_predict=yaml_config['preproc']['months_to_predict'],
                     define_churn_days=yaml_config['preproc']['define_churn_days'],
                     pca_components=yaml_config['preproc']['pca_components'])
