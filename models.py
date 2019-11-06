import xgboost as xgb
import lightgbm as lgb

def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.01,
        'max_depth': 4,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'alpha': 1,
        'random_state': 42,
        'silent': True
    }

    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    model_xgb = xgb.train(params,
                          tr_data,
                          15000,
                          watchlist,
                          maximize=False,
                          early_stopping_rounds=100,
                          verbose_eval=100)

    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = model_xgb.predict(dtest,
                                   ntree_limit=model_xgb.best_ntree_limit)

    return xgb_pred_y, model_xgb


# LightGBM Regressor


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 40,
        "learning_rate": 0.001,
        "bagging_fraction": 0.6,
        "feature_fraction": 0.6,
        "bagging_frequency": 6,
        "bagging_seed": 42,
        "verbosity": -1,
        "seed": 42
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params,
                      lgtrain,
                      15000,
                      valid_sets=[lgtrain, lgval],
                      early_stopping_rounds=100,
                      verbose_eval=150,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result


