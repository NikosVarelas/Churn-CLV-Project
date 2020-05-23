import pickle
import os
import lightgbm as lgb

from churnclv import BASE_PATH


def main():
    print('Loading Data')
    with open(BASE_PATH + '/output/datasets.pickle', 'rb') as handle:
        data = pickle.load(handle)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0,
        "max_depth": 8,
        "num_leaves": 128,
        "max_bin": 512,
        "num_iterations": 100000,
        "n_estimators": 1000
    }
    lgb_train = lgb.Dataset(data['x_train_clv'].values,
                            label=data['y_train_clv'].values)
    lgb_val = lgb.Dataset(data['x_val_clv'].values,
                          label=data['y_val_clv'].values)

    model = lgb.train(params,
                      lgb_train,
                      valid_sets=lgb_val,
                      num_boost_round=5000,
                      early_stopping_rounds=1000)

    if not os.path.isdir(BASE_PATH + '/trained_models'):
        os.mkdir(BASE_PATH + '/trained_models')
    model.save_model(BASE_PATH + '/trained_models/' + 'lgb_model.h5')


if __name__ == '__main__':
    main()
