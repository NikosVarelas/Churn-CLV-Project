import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta

from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from scipy.stats import entropy, skew, kurtosis, mode

# Helper functions


class Preprocessor(object):
    def __init__(self, data, months, date_col, basket_col, churn_days):
        """

        :param data: Pandas dataframe
        :param months: Integer value indicates how many months we will predict ahead
        """
        self.data = data
        self.months = months
        self.date_col = date_col
        self.basket_col = basket_col
        self.churn_days = churn_days
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
        basket_df = data.groupby([key, self.date_col, self.basket_col])\
            [value].sum().reset_index()
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

    def think(self, data, key, item_net_value):
        basket_df = self.basket_calc(data, key, item_net_value)
        visits_df = self.days_btw_calc(data, key)
        basket_features = self.compute_stats(basket_df, key, 'basket')
        visits_features = self.compute_stats(visits_df, key, 'visits')
        output = self.merge_dfs([basket_features, visits_features], key)

        return output

    def  fit(self, key, item_net_value):
        train_features = self.think(self.train, key, item_net_value)
        predict_features = self.think(self.predict, key, item_net_value)
        return train_features, predict_features


class LaggedFeatures(Preprocessor):
    def __init__(self, data, months, date_col, basket_col, lag):
        super(Preprocessor, self).__init__(data, months, date_col, basket_col)
        self.lag = lag

    def month_events_lagged(self, key):
        filtering_date = self.max_date - relativedelta(months=self.lag)
        last_events = self.data.loc[self.data[self.date_col] > filtering_date]
        last_events = last_events.groupby(key)[
            self.basket_col].nunique().reset_index()
        output = last_events.rename(
            columns={self.basket_col: 'events_lag' + str(self.lag)})

        return output.drop(self.date_col, axis=1)

    def month_amount_lagged(self, key, value):
        filtering = self.max_date - relativedelta(months=self.lag)
        last_amount = self.data.groupby(key)[value].sum().reset_index()
        last_amount = last_amount[filtering]
        last_amount = last_amount.rename(
            columns={value: 'amount_lag' + str(self.lag)})

        return last_amount.drop(self.date_col, axis=1)

    def fit(self, key, value):
        output = self.month_events_lagged(key, value)
        output = self.month_amount_lagged(key, value)
        return output


class LabelEstimator(Preprocessor):
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

    def fit(self, key, value):
        clv = self.clv_estimate(key, value)
        churn = self.churn_estimate(key)
        output = clv.merge(churn, on=key)
        return output[['Customer_no', 'churn', 'clv']]


def preprocessing_basket(data):
    # Total amount spend,mean of baskets,std of baskets,min,max and skewness basket

    basket_df = data.groupby(['Customer_no', 'Basket_id'
                              ]).item_net_amount.sum().reset_index()

    amount_df = basket_df.groupby('Customer_no').item_net_amount.agg([
        np.mean, np.median, np.std, np.min, np.max, 'sum', skew, kurtosis,
        lambda x: mode(x)[0][0]
    ])
    # Finding how many different products bought by customers

    products_bought = data.groupby('Customer_no').EAN.nunique().reset_index()
    products_bought = products_bought.rename(columns={'EAN': 'diff_products'})

    amount_df = amount_df.merge(products_bought, on='Customer_no')

    return amount_df


def entropy_calc(data, key, value, name):
    """

    :param data: Pandas dataframe
    :param key: String key to perform the aggregation
    :param value: String the column to perform the aggregation
    :param name: The name of the column
    :return: Pandas dataframe
    """
    ent_df = data.groupby(key)[value].nunique()
    ent_df = ent_df.groupby(
        key[0]).apply(lambda x: entropy(x / sum(x))).reset_index()
    output = ent_df.rename(columns={value: name})
    return output


class PcaModel(object):
    """

    Performs pca method
    """
    def __init__(self, method):
        """

        :type method: PCA method from sklearn
        """
        self.method = method

    def fit(self, df):
        """

        :param df: Pandas dataframe
        :return: Pandas dataframe trasformed by the pca method
        """
        self.method = self.method.fit(df)
        output = self.method.transform(df)
        output = pd.DataFrame(
            data=output,
            columns=['pc_' + str(x + 1) for x in range(output.shape[1])],
            index=df.index)

        print('Variance explained by components:\n {}\n'.format(
            self.method.explained_variance_ratio_))
        print('Total variance explained:\n{}\n'.format(
            sum(self.method.explained_variance_ratio_)))
        print('Singular values:\n{}'.format(self.method.singular_values_))
        return output

    def transformer(self, df):
        """

        :param df: Pandas dataframe
        :return: Pandas dataframe transformed by the pca input method
        """
        output = self.method.transform(df)

        output = pd.DataFrame(
            data=output,
            columns=['pc_' + str(x + 1) for x in range(output.shape[1])],
            index=df.index)

        return output


class Normalization:
    def __init__(self, scaler):
        """

        :param scaler: Sklearn scaler (e.g. StandardScaler)
        """
        self.scaler = scaler

    def normaliser_fit(self, df):
        """

        :param df: Pandas dataframe
        :return: Returns the input dataframe normalised and the normalised method
        """
        self.scaler = self.scaler.fit(df)
        output = self.scaler.transform(df)
        output = pd.DataFrame(output)

        output.columns = df.columns
        return output

    def normaliser_transform(self, df):
        """

        :param df: Pandas dataframe
        :return: Pandas dataframe transformed based on scaler
        """
        output = self.scaler.transform(df)
        output = pd.DataFrame(output)
        output.columns = df.columns

        return output


def get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    """

    :param df: Pandas dataframe
    :param n: Integer indicating the number of top correlation pairs to be returned
    :return: Tuple of correlations
    """
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def train_valid_test_split(df, perc, target):
    """

    :param df: Pandas dataframe
    :param perc: (0,1) The percentage of the validation test split
    :param target: String Target value to be excluded from the split
    :return:
    """
    x = df.loc[:, df.columns != target]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        stratify=y,
                                                        test_size=perc)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=perc)

    return x_train, x_test, x_val, y_train, y_val, y_test


def random_forest_importances(df, target):
    x = df.loc[:, df.columns != target]
    x = x.drop('Customer_no', axis=1)
    y = df[target]

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(x, y)

    important_features = pd.Series(data=clf.feature_importances_,
                                   index=x.columns)

    return important_features.sort_values(ascending=False)
