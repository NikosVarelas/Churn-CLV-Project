from churnclv.utils.preproc import LabelCalculator, LaggedFeatures
# from churnclv.utils.feature_engineering import PcaModel, Normalization


class Pipeline(LabelCalculator, LaggedFeatures):
    def __init__(self,
                 data,
                 months,
                 date_col,
                 basket_col,
                 churn_days,
                 lag):
        LaggedFeatures.__init__(self, data, months, date_col, basket_col, lag)
        LabelCalculator.__init__(self, data, months, date_col, basket_col,
                                 churn_days)
        self.train_df = None
        self.predict_df = None
        self.lagged_train = None
        self.lagged_predict = None
        self.labels = None

    def fit(self, key, value):
        self.train_df, self.predict_df = super(LaggedFeatures, self).transform(key, value)
        self.lagged_train, self.lagged_predict = LaggedFeatures.transform(self, key, value)
        self.labels = LabelCalculator.transform(self, key, value)

        return self

    @staticmethod
    def left_join(left, right, key):
        joined_table = left.merge(right, on=key)
        output = joined_table.fillna(0)

        return output

    def transform(self, key, value):
        train = self.left_join(self.train_df, self.lagged_train, key)
        train_set = self.left_join(train, self.labels, key)
        predict_set = self.left_join(self.predict_df, self.lagged_predict, key)

        return train_set, predict_set










