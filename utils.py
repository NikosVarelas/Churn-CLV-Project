import pandas as pd
import numpy as np
import datetime
import os
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense

from scipy.stats import entropy,skew,mode,kurtosis,boxcox,mode,pearsonr

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline,Pipeline

import inspect
import xgboost as xgb
import lightgbm as lgb


# Helper functions

def missing_values(data):
    # getting the sum of null values and ordering
    total = data.isnull().sum().sort_values(ascending = False) 
    #getting the percent and order of null
    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(
            ascending = False)
    # Concatenating the total and percent
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
    return df.loc[~(df['Total'] == 0)] 


def count_zeros(df):
    count = sum((df==0))
    return count


def preprocessing_data(data,date,prediction=False):
    
    if prediction == False:
        filter_day = data['Transaction_date'].max()-relativedelta(months=date)
        filter_train = data['Transaction_date'] < filter_day
        filter_labels = data['Transaction_date'] >= filter_day
        
        # Obtaining the labels of CLV or churn
        CLV = data[filter_labels].groupby('Customer_no').item_net_amount.sum().reset_index()
        CLV = CLV.rename(columns = {'item_net_amount':'CLV'})
         
        last_p = data['Transaction_date'].max()-data.groupby('Customer_no').Transaction_date.max()
        last_p = last_p.apply(lambda x: x.days)
        churn = (last_p>14).astype(int).reset_index()
        churn = churn.rename(columns = {'Transaction_date':'churn'})
            
    else:
        filter_day = data['Transaction_date'].min()+relativedelta(months=date)
        filter_train = data['Transaction_date'] >= filter_day
    
    # first last order and number of active days
    
    data = data[filter_train]
    end_time=np.datetime64(data['Transaction_date'].max())
    begin_time=np.datetime64(data['Transaction_date'].min())

    customer_df=data.groupby('Customer_no',as_index=False).Transaction_date.agg({np.max,np.min,np.ptp}).reset_index()
   
    customer_df['last_purchase']=end_time-customer_df['amax']
    customer_df['first_purchase']=customer_df['amin']-begin_time

    customer_df['ptp']=customer_df['ptp'].apply(lambda x: x.days)
    customer_df['last_purchase']=customer_df['last_purchase'].apply(lambda x: x.days)
    customer_df['first_purchase']=customer_df['first_purchase'].apply(lambda x: x.days)

    customer_df=customer_df.rename(columns={'ptp':'active_days'})
    customer_df=customer_df.drop(['amax','amin'],axis=1)
    
    # Visits of each customer

    frequency_df=data.groupby('Customer_no').Basket_id.nunique().reset_index()
    frequency_df=frequency_df.rename(columns={'Basket_id':'events'})

    customer_df=customer_df.merge(frequency_df,on='Customer_no')

    # Total amount spend,mean of baskets,std of baskets,min,max and skewness basket

    basket_df=data.groupby(['Customer_no','Basket_id']).item_net_amount.sum().reset_index()

    amount_df=basket_df.groupby('Customer_no').item_net_amount.agg([np.mean,np.median,np.std,np.min,np.max,'sum',skew,
                                                                    kurtosis,lambda x: mode(x)[0][0]])
    amount_df=amount_df.rename(columns={'sum':'amount',
                                        'mean':'mean_basket',
                                        'median':'median_basket',
                                        'std':'std_basket',
                                        'amin':'min_basket',
                                        'amax':'max_basket',
                                        'skew':'skew_basket',
                                        'kurtosis':'kur_basket',
                                       '<lambda>':'mode_basket'})
    
    customer_df=customer_df.merge(amount_df,on='Customer_no')
    
    # Finding how many different products bought by customers

    products_bought=data.groupby('Customer_no').EAN.nunique().reset_index()
    products_bought=products_bought.rename(columns={'EAN':'diff_products'})

    customer_df=customer_df.merge(products_bought,on='Customer_no')

    # Statistics of days between visits
    
    kur = data[['days_btw_visits','Customer_no']].dropna()
    
    stats_of_btw_days=kur.groupby('Customer_no').days_btw_visits.agg([np.mean,np.median,np.std,np.min,np.max,skew,
                                                                      kurtosis,lambda x: mode(x)[0][0],count_zeros])
    stats_of_btw_days=stats_of_btw_days.rename(columns={'mean':'mean_btv',
                                                        'median':'median_btv',
                                                        'std':'std_btv',
                                                        'amin':'min_btv',
                                                        'amax':'max_btv',
                                                        'skew':'skew_btv',
                                                        'kurtosis':'kur_btv',
                                                        '<lambda>':'mode_btv'})

    customer_df=customer_df.merge(stats_of_btw_days,on='Customer_no')
    
    # Customers spendings over time
    
    customer_df['spends_mean_time']=customer_df['mean_basket']/customer_df['mean_btv']
    customer_df['spends_std_time']=customer_df['std_basket']/customer_df['std_btv']
    customer_df['purchase_mean']=customer_df['last_purchase']/customer_df['mean_btv']
    
    # Store entropy
    
    store_entropy = data.groupby(['Customer_no','Store_number']).Basket_id.nunique()
    store_entropy = store_entropy.groupby('Customer_no').apply(lambda x:entropy(x/sum(x))).reset_index()
    store_entropy = store_entropy.rename(columns={'Basket_id':'store_entr'})
    
    # Month entropy
    
    month_entropy = data.groupby(['Customer_no','Transaction_date_month']).Basket_id.nunique()
    month_entropy = month_entropy.groupby('Customer_no').apply(lambda x:entropy(x/sum(x))).reset_index()
    month_entropy = month_entropy.rename(columns={'Basket_id':'month_entr'})
    
    # Day entropy
    
    day_entropy = data.groupby(['Customer_no','dayofWeek']).Basket_id.nunique()
    day_entropy = day_entropy.groupby('Customer_no').apply(lambda x:entropy(x/sum(x))).reset_index()
    day_entropy = day_entropy.rename(columns={'Basket_id':'day_entr'})
    
    customer_df = customer_df.merge(store_entropy,on='Customer_no')
    customer_df = customer_df.merge(month_entropy,on='Customer_no')
    customer_df = customer_df.merge(day_entropy,on='Customer_no')
    
    # customers last month's amount
    
    one_month = relativedelta(months=1)
    filter_month = data['Transaction_date'] >= (data['Transaction_date'].max()-one_month)
    
    last_am = data[filter_month].groupby('Customer_no').item_net_amount.sum().reset_index()
    last_am = last_am.rename(columns = {'item_net_amount':'last_amount'})
    
    customer_df = customer_df.merge(last_am,on = 'Customer_no', how='left')
    
    # customers last month's visits
    
    filter_month = data['Transaction_date'] >= (data['Transaction_date'].max()-one_month)
    
    last_ev = data[filter_month].groupby('Customer_no').Basket_id.nunique().reset_index()
    last_ev = last_ev.rename(columns = {'Basket_id':'last_events'})
    
    customer_df = customer_df.merge(last_ev,on = 'Customer_no', how='left')
    
    if prediction == False:
        customer_df = customer_df.merge(CLV,on = 'Customer_no', how='left')
        customer_df = customer_df.merge(churn,on = 'Customer_no', how='left')
                
    customer_df = customer_df.fillna(0)
    
    return customer_df


def pca_fit(df,comp):
    
    pca = PCA(n_components=comp)
    pca.fit(df)
    
    print('Variance explained by components:\n {}\n'.format(pca.explained_variance_ratio_))  
    print('Total variance explained:\n{}\n'.format(sum(pca.explained_variance_ratio_)))
    print('Singular values:\n{}'.format(pca.singular_values_))  


def pca_transformer(X,Y,Z,K,comp):
    
    pca = PCA(n_components=comp)
    train = pca.fit_transform(X)
    val = pca.transform(Y)
    test = pca.transform(Z)
    predict = pca.transform(K)

    train = pd.DataFrame(data = train
                            , columns = ['pc1', 'pc2', 'pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10']
                            , index = X.index)
    val = pd.DataFrame(data = val
                            , columns = ['pc1', 'pc2', 'pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10']
                            , index = Y.index)
    test = pd.DataFrame(data = test
                            , columns = ['pc1', 'pc2', 'pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10']
                            , index = Z.index)
    predict = pd.DataFrame(data = predict
                            , columns = ['pc1', 'pc2', 'pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10']
                            , index = K.index)
    
    return train , val , test , predict



def input_normaliser(X,Y,Z,K):
    scaler=StandardScaler()
    scaler.fit(X)
    
    train=scaler.transform(X)
    valid=scaler.transform(Y)
    test=scaler.transform(Z)
    predict=scaler.transform(K)
    
    train=pd.DataFrame(train)
    valid=pd.DataFrame(valid)
    test=pd.DataFrame(test)
    predict=pd.DataFrame(predict)
    
    train.columns=X.columns
    valid.columns=X.columns
    test.columns=X.columns
    predict.columns=X.columns
    return train , valid , test , predict


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def train_valid_test_split(df,perc,target):
    
    X = df.loc[:, df.columns != target]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=perc)

    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train,stratify=y_train, test_size=perc)
    
    return X_train, X_test , X_val , y_train , y_val , y_test


# XGBoost Classifier

def run_xgb(train_X, train_y, val_X, val_y, test_X, test_y):
    params = {'objective': 'binary:logistic', 
          'eval_metric': 'logloss',
          'eta': 0.01,
          'max_depth': 4, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':1,
          'random_state': 42, 
          'silent': True}
    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 15000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit)
    
    return xgb_pred_y, model_xgb


# LightGBM Regressor

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "binary",
        "metric" : "binary_logloss",
        "num_leaves" : 40,
        "learning_rate" : 0.001,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 15000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    
    return pred_test_y, model, evals_result


def random_forest_importances(df,target):
    
    X = df.loc[:, df.columns != target]
    X = X.drop('Customer_no', axis = 1)
    y = df[target]
    
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X,y)

    important_features = pd.Series(data=clf.feature_importances_,index=X.columns)
    
    return important_features.sort_values(ascending=False)
