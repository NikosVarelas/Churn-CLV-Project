from utils import *

if __name__ == "__main__":
    print("Reading Data")
    my_data = pd.read_csv('./data/customerData.csv', sep=',')
    my_data['Transaction_date'] = pd.to_datetime(my_data['Transaction_date'])

    days = my_data.sort_values(
        by=['Transaction_date']).drop_duplicates('Basket_id')
    my_data['previous_visit'] = days.groupby(
        'Customer_no').Transaction_date.shift()
    my_data['days_btw_visits'] = my_data['Transaction_date'] - my_data[
        'previous_visit']

    my_data['days_btw_visits'] = my_data['days_btw_visits'].apply(
        lambda x: x.days)

    # Predicting CLV one month ahead

    start = timer()
    print("Started preproc")
    customers_train = preprocessing_data(my_data, 1)
    customers_predict = preprocessing_data(my_data, 1, prediction=True)

    # Dropping customers with less than 5 events

    filter_events = customers_train['events'] > 4
    customers_train = customers_train[filter_events]

    end = timer()
    print("Time took for preproc: " + str(end - start))

    # Dropping features with low feature importance

    X = customers_train.drop([
        'min_btv', 'mode_btv', 'store_entr', 'Customer_no', 'skew_btv',
        'kur_btv', 'CLV', 'max_basket', 'count_zeros'
    ],
                             axis=1)

    X_pred = customers_predict.drop([
        'min_btv', 'mode_btv', 'store_entr', 'Customer_no', 'skew_btv',
        'kur_btv', 'max_basket', 'count_zeros'
    ],
                                    axis=1)

    # Train validation test split

    X_train, X_test, X_val, y_train, y_val, y_test = train_valid_test_split(
        X, 0.3, 'churn')

    # Normalising the sets

    X_train, X_val, X_test, X_pred = input_normaliser(X_train, X_val, X_test,
                                                      X_pred)

    # Adding the components of pca to the sets

    X_train, X_val, X_test, X_pred = pca_transformer(X_train, X_val, X_test,
                                                     X_pred, 10)

    #Train LightGBM

    pred_test, model, evals_result = run_lgb(X_train, y_train, X_val, y_val,
                                             X_test)
    print("LightGBM Training Completed...")

    # Training XGB

    pred_test_xgb, model_xgb = run_xgb(X_train, y_train, X_val, y_val, X_test,
                                       y_test)
    print("XGB Training Completed...")

    X = xgb.DMatrix(X_pred)
    xgb_pred = model_xgb.predict(X, ntree_limit=model_xgb.best_ntree_limit)

    reg = l1_l2(l1=0, l2=0.01)

    model_log = Sequential()
    model_log.add(
        Dense(
            1,
            activation='sigmoid',
            W_regularizer=reg,
            input_dim=X_train.shape[1]))
    model_log.compile(optimizer='Adam', loss='binary_crossentropy')
    model_log.fit(
        X_train, y_train, nb_epoch=700, validation_data=(X_val, y_val))

    prob3 = model_log.predict_proba(X_pred)
    prob4 = model_log.predict_proba(X_test)

    #Checking how well calibrated are the probas obtained by Logistic Classifier

    X_lr = X_train
    X_lr = X_lr.append(X_val)
    X_lr = X_lr.append(X_test)
    y_lr = y_train
    y_lr = y_lr.append(y_val)
    y_lr = y_lr.append(y_test)

    probs = model_log.predict_proba(X_lr)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_lr, probs, n_bins=10)

    fig, ax = plt.subplots(1, figsize=(12, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-')
    plt.plot([0, 1], [0, 1], '--', color='gray')

    sns.despine(left=True, bottom=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.title("$Logistic$ Calibration Curve", fontsize=20)

    #AUC Results
    
    print("LightGBM AUC: " + str(roc_auc_score(y_test, pred_test)))
    print("XGBoost AUC: " + str(roc_auc_score(y_test, pred_test_xgb)))
    print("Logistic Regression AUC: " + str(roc_auc_score(y_test, prob4)))
