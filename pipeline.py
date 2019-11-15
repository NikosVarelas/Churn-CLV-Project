from utils import LabelCalculator, LaggedFeatures
from feature_engineering import PcaModel, Normalization


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


