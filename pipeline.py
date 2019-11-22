from churnclv.utils.preproc import LabelCalculator, LaggedFeatures
from churnclv.utils.feature_engineering import PcaModel, Normalization


class Pipeline(LabelCalculator, LaggedFeatures, PcaModel, Normalization):
    def __init__(self,
                 data,
                 months,
                 date_col,
                 basket_col,
                 churn_days,
                 lag,
                 n_components=None,
                 normalisation=True):
        LaggedFeatures.__init__(self, data, months, date_col, basket_col, lag)
        LabelCalculator.__init__(self, data, months, date_col, basket_col,
                                 churn_days)
        PcaModel.__init__(self, n_components)
        self.normalisation = normalisation
        self.train_df = None
        self.predict_df = None
        self.lagged_train = None
        self.lagged_predict = None
        self.labels = None

    def compute_tables(self, key, value):
        self.train_df, self.predict_df = super(LaggedFeatures, self).transform(key, value)
        self.lagged_train, self.lagged_predict = LaggedFeatures.transform(self, key, value)
        self.labels = LabelCalculator.transform(key, value)
        return self

    @staticmethod
    def left_join(left, right, key):
        joined_table = left.merge(right, on=key)
        output = joined_table.fillna(0)

        return output

    def join_tables(self, key, value):
        self.compute_tables(key, value)
        train = self.left_join(self.train_df, self.lagged_train, key)
        train_set = self.left_join(train, self.lagged_train, key)
        predict_set = self.left_join(self.predict_df, key)

        return train_set, predict_set










