import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PcaModel(PCA):
    """

    Performs pca method
    """

    def fit(self, df, y=None):
        """

        :param df:
        :param y:
        :return:
        """
        self._fit(df)
        print('Variance explained by components:\n {}\n'.format(
            self.explained_variance_ratio_))
        print('Total variance explained:\n{}\n'.format(
            sum(self.explained_variance_ratio_)))
        print('Singular values:\n{}'.format(self.singular_values_))
        return self

    def transform(self, df):
        """

        :param df: Pandas dataframe
        :return: Pandas dataframe transformed by the pca input method
        """
        output = PCA.transform(self, df)
        output = pd.DataFrame(
            data=output,
            columns=['pc_' + str(x + 1) for x in range(self.n_components)],
            index=df.index)

        return output


class Normalization(StandardScaler):

    def transform(self, df, copy=None):
        """

        :param df: Pandas dataframe
        :return: Pandas dataframe transformed based on scaler
        """
        output = StandardScaler.transform(self, df)
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

