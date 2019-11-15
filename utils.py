import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from functools import reduce
from scipy.stats import skew, kurtosis, mode

# Helper functions


class Preprocessor(object):
    def __init__(self, data, months, date_col, basket_col):
        """

        :param data: Pandas dataframe
        :param months: Integer value indicates how many months we will predict ahead
        """
        self.data = data
        self.months = months
        self.date_col = date_col
        self.basket_col = basket_col
        self.max_date, self.min_date = self.min_max_date()
        self.train = self.train_filter()
        self.predict = self.predict_filter()
        self.label = self.label_filter()

    def min_max_date(self):
        """

        :return: The min and max of the column
        """
        max_date = self.data[self.date_col].max()
        min_date = self.data[self.date_col].min()
        return max_date, min_date

    def train_filter(self):
        """

        :return: Pandas dataframe
        """
        filtering_date = self.max_date - relativedelta(months=self.months)
        output = self.data.loc[self.data[self.date_col] <= filtering_date]
        return output

    def predict_filter(self):
        filtering_date = self.min_date + relativedelta(months=self.months)

        output = self.data.loc[self.data[self.date_col] >= filtering_date]
        return output

    def label_filter(self):
        filtering_date = self.max_date - relativedelta(months=self.months)
        output = self.data.loc[self.data[self.date_col] > filtering_date]
        return output

    def days_btw_calc(self, data, key):
        days_df = data.sort_values(by=[self.date_col]).drop_duplicates(
            self.basket_col)
        data['previous_visit'] = days_df.groupby(key).Transaction_date.shift()
        output = data
        output['visits'] = data[self.date_col] - data['previous_visit']

        output['visits'] = output['visits'].apply(lambda x: x.days)
        return output.dropna()

    def basket_calc(self, data, key, value):
        basket_df = data.groupby([key, self.date_col,
                                  self.basket_col])[value].sum().reset_index()
        basket_df = basket_df.rename(columns={value: 'basket'})

        return basket_df

    @staticmethod
    def agg_stats(data, key, value):
        """

        :param data: Pandas dataframe
        :param key: List of StringsKey to perform the aggregation
        :param value: String The column which will compute the stats
        :return: A pandas dataframe aggregated
        """
        output = data.groupby(key)[value].agg({
            'mean_' + value:
            np.mean,
            'median_' + value:
            np.median,
            'std_' + value:
            np.std,
            'min_' + value:
            np.min,
            'max_' + value:
            np.max,
            'sum_' + value:
            'sum',
            'skew_' + value:
            skew,
            'kurtosis_' + value:
            kurtosis,
            'mode_' + value:
            lambda x: mode(x)[0][0],
            'unique_' + value:
            lambda x: x.nunique()
        }).reset_index()

        return output

    @staticmethod
    def ma_calc(data, key, value, period):
        output = data.groupby(key).rolling(period)[value].mean().reset_index()
        output = output.groupby(key)[value].apply(
            lambda x: x.tail(1)).reset_index()
        output = output.rename(
            columns={value: 'ma' + str(period) + '_' + value})
        output = output.drop('level_1', axis=1)
        return output

    @staticmethod
    def ewma_calc(data, key, value, alpha):
        data['ewma'] = data.groupby(key)[value].apply(
            lambda x: x.ewm(alpha=alpha).mean())
        output = data.groupby(key)[value].apply(
            lambda x: x.tail(1)).reset_index()
        output = output.rename(
            columns={value: 'ewma' + str(alpha) + '_' + value})
        output = output.drop('level_1', axis=1)
        return output

    @staticmethod
    def merge_dfs(list_dfs, key):
        """

        :param list_dfs: a list of dataframes
        :param key: String the key that the join will be performed
        :return: Pandas dataframe
        """
        output = reduce(lambda left, right: pd.merge(left, right, on=key),
                        list_dfs)

        return output

    def compute_stats(self, data, key, value):
        """

        :param data: Pandas dataframe
        :param key: List of strings to perform aggregation
        :param value: String The column which will compute the stats
        :return: Pandas dataframe
        """
        stats = self.agg_stats(data, key, value)
        ewma_09 = self.ewma_calc(data, key, value, 0.9)
        ewma_05 = self.ewma_calc(data, key, value, 0.5)
        ma_5 = self.ma_calc(data, key, value, 5)
        ma_3 = self.ma_calc(data, key, value, 3)
        output = self.merge_dfs([stats, ewma_05, ewma_09, ma_3, ma_5], key)

        return output

    def stats_visits_baskets(self, data, key, item_net_value):
        basket_df = self.basket_calc(data, key, item_net_value)
        visits_df = self.days_btw_calc(data, key)
        basket_features = self.compute_stats(basket_df, key, 'basket')
        visits_features = self.compute_stats(visits_df, key, 'visits')
        output = self.merge_dfs([basket_features, visits_features], key)

        return output

    def transform(self, key, value):
        train_features = self.stats_visits_baskets(self.train, key, value)
        predict_features = self.stats_visits_baskets(self.predict, key, value)
        return train_features, predict_features


class LaggedFeatures(Preprocessor):
    def __init__(self, data, months, date_col, basket_col, lag):
        super(LaggedFeatures, self).__init__(data, months, date_col, basket_col)
        self.lag = lag
        self.train_date_filter = self.max_date - relativedelta(
            months=self.lag + self.months)
        self.predict_date_filter = self.max_date - relativedelta(
            months=self.lag)

    def month_events_lagged(self, data, key, filtering_date):
        last_events = data.loc[data[self.date_col] > filtering_date]
        last_events = last_events.groupby(key)[
            self.basket_col].nunique().reset_index()
        output = last_events.rename(
            columns={self.basket_col: 'events_lag' + str(self.lag)})

        return output

    def month_amount_lagged(self, data, key, value, filtering_date):
        last_amount = data.loc[data[self.date_col] > filtering_date]
        last_amount = last_amount.groupby(key)[value].sum().reset_index()
        last_amount = last_amount.rename(
            columns={value: 'amount_lag' + str(self.lag)})

        return last_amount

    def transform(self, key, value):
        train_events = self.month_events_lagged(self.train, key,
                                                self.train_date_filter)
        train_amount = self.month_amount_lagged(self.train, key, value,
                                                self.train_date_filter)
        predict_events = self.month_events_lagged(self.predict, key,
                                                  self.predict_date_filter)
        predict_amount = self.month_amount_lagged(self.predict, key, value,
                                                  self.predict_date_filter)
        train_lagged_features = self.merge_dfs([train_events, train_amount],
                                               key)
        predict_lagged_features = self.merge_dfs(
            [predict_events, predict_amount], key)
        return train_lagged_features, predict_lagged_features


class LabelCalculator(Preprocessor):
    def __init__(self, data, months, date_col, basket_col, churn_days):
        Preprocessor.__init__(self, data, months, date_col, basket_col)
        self.churn_days = churn_days

    def clv_estimate(self, key, value):
        output = self.label.groupby(key)[value].sum().reset_index()
        output = output.rename(columns={value: 'clv'})
        return output

    def churn_estimate(self, key):
        label_df = self.label.groupby(key)[self.date_col].agg(
            {'last_day': np.max})
        label_df['last_pur'] = self.max_date - label_df['last_day']
        label_df['last_pur'] = label_df['last_pur'].apply(lambda x: x.days)
        label_df['churn'] = np.where(label_df['last_pur'] >= self.churn_days,
                                     1, 0)

        return label_df

    def transform(self, key, value):
        clv = self.clv_estimate(key, value)
        churn = self.churn_estimate(key)
        output = clv.merge(churn, on=key)
        return output[['Customer_no', 'churn', 'clv']]
