from churnclv.utils.preproc import LabelCalculator, LaggedFeatures
from churnclv.utils.feature_engineering import PcaModel, Normalization, train_valid_test_split


class Pipeline(LabelCalculator, LaggedFeatures):
    def __init__(self, data, months, date_col, basket_col, churn_days, lag, normalisation=True, n_components=None):
        LabelCalculator.__init__(self, data, months, date_col, basket_col,
                                 churn_days)
        LaggedFeatures.__init__(self, data, months, date_col, basket_col, lag)
        self.train_df = None
        self.predict_df = None
        self.lagged_train = None
        self.lagged_predict = None
        self.labels = None
        self.normalisation = normalisation
        self.n_components = n_components

    def fit(self, key, value):
        self.train_df, self.predict_df = super(LaggedFeatures,
                                               self).transform(key, value)
        self.lagged_train, self.lagged_predict = LaggedFeatures.transform(
            self, key, value)
        self.labels = LabelCalculator.transform(self, key, value)

        return self

    @staticmethod
    def left_join(left, right, key):
        joined_table = left.merge(right, on=key)
        output = joined_table.fillna(0)

        return output

    def create_sets(self, key):
        train = self.left_join(self.train_df, self.lagged_train, key)
        train_set = self.left_join(train, self.labels, key)
        predict_set = self.left_join(self.predict_df, self.lagged_predict, key)

        return train_set, predict_set

    def transform(self, key, train_set, predict_set, target):
        train_set = train_set.drop(key, axis=1)
        x_pred = predict_set.drop(key, axis=1)
        if target == 'churn':
            x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(
                train_set.drop('clv', axis=1), target, 0.3, stratify_fold=True)
        else:
            x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(
                train_set.drop('churn', axis=1), target, 0.3, stratify_fold=False)
        if self.normalisation:
            norm = Normalization()
            norm.fit(x_train)
            x_train = norm.transform(x_train)
            x_val = norm.transform(x_val)
            x_test = norm.transform(x_test)
            x_pred = norm.transform(x_pred)
        if self.n_components is not None:
            pca = PcaModel(n_components=self.n_components)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_val = pca.transform(x_val)
            x_test = pca.transform(x_test)
            x_pred = pca.transform(x_pred)

        return x_train, x_val, x_test, x_pred, y_train, y_val, y_test
