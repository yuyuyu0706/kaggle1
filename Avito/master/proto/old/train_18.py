import time
import gc
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle as pkl
import keras as ks
import tensorflow as tf
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split, KFold
from sklearn.linear_model import Ridge, ElasticNet
from catboost import CatBoostRegressor
 
# Original
from load_data import load_train_data, load_test_data
from load_data import load_train_act_data
from grid_data import xgb_gs, lgb_gs
from set_logger import logger

# TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

ROW = 10000
DIR = './result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
XGBFMAP = DIR + 'xgb.fmap'
TRN_PRED_FILE = DIR + 'train_act_0000_pred_lgbm.csv'
SEED = 42
NFOLDS = 5

def get_col(col_name):
    return lambda x: x[col_name]

def rmse(label_y, pred_X):
    return np.sqrt(mean_squared_error(label_y, pred_X))

def create_feats_map(features):
    with open(XGBFMAP, 'w') as f:
        i = 0
        for feat in features:
            f.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

def cleanName(text):
    try:
        textProc = text.lower()
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except:
        return "name error"

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, tst_series=None, target=None, 
                  min_samples_leaf=1, smoothing=1, noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 
    ave = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(ave["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    ave[target.name] = prior * (1 - smoothing) + ave["mean"] * smoothing
    ave.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(trn_series.to_frame(trn_series.name),
                             ave.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
                             on=trn_series.name,
                             how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(tst_series.to_frame(tst_series.name),
                             ave.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
                             on=tst_series.name,
                             how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

def ens_ridge(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    ridge_params = {
        'alpha': 20.0,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'max_iter': None,
        'tol': 0.001,
        'solver': 'auto',
        'random_state': 42
        }
    model = Ridge(**ridge_params)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X[trn_i]
        y_trn = trn_y[trn_i]
        x_test = trn_X[test_i]

        model.fit(x_trn, y_trn)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - Ridge Train: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def ens_en(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    en_params = {
        'alpha': 1.0
        }
    model = ElasticNet(**en_params)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X[trn_i]
        y_trn = trn_y[trn_i]
        x_test = trn_X[test_i]

        model.fit(x_trn, y_trn)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - ElasticNet Train: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def ens_nn(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    lr = 0.05
    bz = 100000
    ep = 500
    op = ks.optimizers.Adam(lr=lr)
    early = ks.callbacks.EarlyStopping(monitor='val_loss', patience=500, mode='min')

    logger.info('NN Train Shape: {}'.format(trn_X.shape))
    logger.info('NN Test Shape : {}'.format(test_X.shape))

    # model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=True)
    model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=False)
    # out = ks.layers.Dense(192, activation='relu')(model_in)
    # out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(16, activation='relu')(model_in)
    out = ks.layers.Dense(8, activation='relu')(out)
    out = ks.layers.Dense(8, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=op)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X.index)))):
        # x_trn = trn_X[trn_i]
        # y_trn = trn_y[trn_i]
        x_trn = trn_X.iloc[trn_i]
        y_trn = trn_y.iloc[trn_i]
        x_trn, x_val, y_trn, y_val = train_test_split(
                                        x_trn, y_trn, test_size=0.10,
                                        shuffle=False, random_state=23)
        # x_test = trn_X[test_i]
        x_test = trn_X.iloc[test_i]

        model.fit(x=x_trn, y=y_trn, validation_data=(x_val, y_val),
                  batch_size=bz, epochs=ep, callbacks=[early], verbose=1)
        pred_trn_X[test_i] = model.predict(x_test, batch_size=bz)[:, 0]
        pred_test_skf[i, :] = model.predict(test_X, batch_size=bz)[:, 0]

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('Rmse - NN Train: {}, lr: {}, bz: {}, ep: {}'.format(rmse_trn, lr, bz, ep))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def ens_cat(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    model = CatBoostRegressor(iterations=1000,
                              learning_rate=0.08,
                              depth=10,
                              eval_metric='RMSE',
                              metric_period=50,
                              calc_feature_importance=True)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X[trn_i]
        y_trn = trn_y[trn_i]
        x_test = trn_X[test_i]

        model.fit(x_trn, y_trn, use_best_model=True)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - CatBoost Train: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def ens_xgb(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    xg_params = {
        "max_depth": 8,           # [4, 6, 8, 12]
        "min_child_weight": 6,    # [4, 6, 8]
        "learning_rate": 0.1,     # [0.05, 0.075, 0.1, 0.2]
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0,
        "num_estimators":100
        }

    model = xgb.XGBRegressor(**xg_params)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X[trn_i]
        y_trn = trn_y[trn_i]
        x_test = trn_X[test_i]

        model.fit(x_trn, y_trn)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('rmse - XGBoost Feature: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def run_nn(trn_X, trn_y, val_X, val_y):
    lr = 0.1
    bz = int(trn_X.shape[0] / 10)
    ep = 50
    op = ks.optimizers.Adam(lr=lr)
    # op = ks.optimizers.SGD(lr=0.001, momentum=0.9)
    # with tf.Session(graph=tf.Graph(), config=config) as sess:
    early = ks.callbacks.EarlyStopping(monitor='loss', patience=0, mode='min')
    
    model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=op)
    model.fit(x=trn_X, y=trn_y, validation_data=(val_X, val_y),
              batch_size=bz, epochs=ep, callbacks=[early], verbose=1)
    
    pred_trn = model.predict(trn_X, batch_size=bz)[:, 0]
    pred_val = model.predict(val_X, batch_size=bz)[:, 0]
    rmse_trn = rmse(trn_y, pred_trn)
    rmse_val = rmse(val_y, pred_val)
    
    logger.info('rmse - Train:{0:.4f} Valid:{1:.4f}'.format(rmse_trn, rmse_val))
    return model

def run_ridge(trn_X, trn_y, val_X, val_y):
    ridge_params = {
        'alpha': 20.0,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'max_iter': None,
        'tol': 0.001,
        'solver': 'auto',
        'random_state': 42
        }

    model = Ridge(**ridge_params)
    model.fit(trn_X, trn_y)
    
    pred_trn = model.predict(trn_X)
    pred_val = model.predict(val_X)
    rmse_trn = rmse(trn_y, pred_trn)
    rmse_val = rmse(val_y, pred_val)
    logger.info('rmse - Train: {}'.format(rmse_trn))
    logger.info('rmse - valid: {}'.format(rmse_val))
    
    return model

def run_cat(trn_X, trn_y, val_X, val_y):

    # cat_params = {
    #    }

    model = CatBoostRegressor(iterations=1000,
                              learning_rate=0.08,
                              depth=10,
                              eval_metric='RMSE',
                              metric_period=50,
                              calc_feature_importance=True)
    # Train Start
    model.fit(trn_X, trn_y,
              eval_set=(val_X, val_y),
              use_best_model=True)

    pred_trn = model.predict(trn_X)
    pred_val = model.predict(val_X)
    rmse_trn = rmse(trn_y, pred_trn)
    rmse_val = rmse(val_y, pred_val)
    logger.info('rmse - Train: {}'.format(rmse_trn))
    logger.info('rmse - valid: {}'.format(rmse_val))
    
    return model

def run_lgb(trn_X, trn_y, val_X, val_y, tfvocab, cat_vars):
    lg_params = {
        "objective": "regression",
        "boosting": "gbdt",
        "metric": "rmse",
        # "max_depth": 15,         # [15]
        "num_leaves": 128,      # [32, 48, 64, 128,,,,32768]
        "learning_rate": 0.07,   # [0.05, 0.07, 0.1, 0.2]
        "feature_fraction": 0.7,
        "bagging_freq": 5,
        "bagging_fraction": 0.7,
        "bagging_seed": 2018,
        "verbosity": -1,
        # "verbose": 0
        }
    
    lg_trn = lgb.Dataset(trn_X, label=trn_y, feature_name=tfvocab, categorical_feature=cat_vars)
    lg_val = lgb.Dataset(val_X, label=val_y, feature_name=tfvocab, categorical_feature=cat_vars)
    
    logger.info('split.train: {}'.format(trn_X.shape))
    logger.info('split.valid: {}'.format(val_X.shape))

    # GridSearch
    # min_params = lgb_gs(lg_params, lg_trn, trn_y, lg_val, val_X, val_y)
   
    # Train Start
    model = lgb.train(lg_params, lg_trn,
                      num_boost_round=16000,
                      valid_sets=[lg_val],
                      early_stopping_rounds=200,
                      verbose_eval=100)
    
    pred_trn = model.predict(trn_X, num_iteration=model.best_iteration)
    pred_val = model.predict(val_X, num_iteration=model.best_iteration)
    rmse_trn = rmse(trn_y, pred_trn)
    rmse_val = rmse(val_y, pred_val)
    logger.info('rmse - Train: {}'.format(rmse_trn))
    logger.info('rmse - valid: {}'.format(rmse_val))

    # Feature Importance
    logger.debug('Feature Importances')
    feat_n = model.feature_name()
    feat_i = list(model.feature_importance())

    df_tmp1 = pd.DataFrame(feat_n, columns={'feat_n'})
    df_tmp2 = pd.DataFrame(feat_i, columns={'feat_i'})
    df_tmp = df_tmp1.join(df_tmp2, how='inner')
    df_tmp = df_tmp.sort_values(by=['feat_i'], ascending=False)
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['feat_i'] = df_tmp['feat_i'] / df_tmp['feat_i'].sum()

    # for i in range(len(df_tmp.index)):
    for i in range(50):
        logger.debug('\t{0:20s} : {1:>10.6f}'.format(
                         df_tmp.ix[i, 0], df_tmp.ix[i, 1]))
    return model

def run_xgb(trn_X, trn_y, val_X, val_y, tfvocab):
    xg_params = {
        "max_depth": 8,           # [4, 6, 8, 12]
        "min_child_weight": 6,    # [4, 6, 8]
        "learning_rate": 0.1,     # [0.05, 0.075, 0.1, 0.2]
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0,
        }

    xg_trn = xgb.DMatrix(trn_X, label=trn_y, feature_names=tfvocab)
    xg_val = xgb.DMatrix(val_X, label=val_y, feature_names=tfvocab)
    watchlist = [(xg_trn, 'train'), (xg_val, 'eval')]
    
    logger.info('split.train: {}'.format(trn_X.shape))
    logger.info('split.valid: {}'.format(val_X.shape))

    # GridSearch
    # min_params = xgb_gs(xg_params, xg_trn, trn_y, xg_val, val_y, wl=watchlist)

    model = xgb.train(xg_params, xg_trn,
                      num_boost_round=5000,
                      evals=watchlist,
                      early_stopping_rounds=100,
                      verbose_eval=50)

    pred_trn = model.predict(xg_trn, ntree_limit=model.best_ntree_limit)
    pred_val = model.predict(xg_val, ntree_limit=model.best_ntree_limit)
    rmse_trn = rmse(trn_y, pred_trn)
    rmse_val = rmse(val_y, pred_val)
    logger.info('rmse - Train: {}'.format(rmse_trn))
    logger.info('rmse - valid: {}'.format(rmse_val))

    # Feature Importance
    # create_feats_map(list(trn_X.columns[2:]))
    feat_i = model.get_fscore(fmap=XGBFMAP)
    
    df_tmp = pd.DataFrame(list(feat_i.items()), columns=['feat_n', 'feat_i'])
    df_tmp = df_tmp.sort_values(by=['feat_i'], ascending=False)
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['feat_i'] = df_tmp['feat_i'] / df_tmp['feat_i'].sum()
    
    # for i in range(len(df_tmp.index)):
    for i in range(15):
        logger.debug('\t{0:20s} : {1:>10.6f}'.format(
                            df_tmp.ix[i, 0], df_tmp.ix[i, 1]))
    return model

if __name__ == '__main__':
    logger.info('Start')

    # temp1_df = load_train_data(nrows=ROW)
    # temp2_df = pd.read_csv('../input/city_population_wiki_v3.csv')
    # train_df = pd.merge(temp1_df, temp2_df, on='city', how='left')
    # del temp1_df, temp2_df
    train_df = load_train_data(nrows=ROW)
    logger.info('Train Data load end {}'.format(train_df.shape))

    test_df = load_test_data(nrows=ROW)
    logger.info('Test load end {}'.format(test_df.shape))

    # test_df = load_period_train_data(nrows=ROW)
    # logger.info('period train load end {}'.format(test_df.shape))

    # pr_test_df = load_period_test_data(nrows=ROW)
    # logger.info('period test load end {}'.format(pr_test_df.shape))

    # test_df = load_train_act_data(nrows=ROW)
    # tmp_df = pd.read_csv(TRN_PRED_FILE, index_col=['item_id'])
    # train_act_df = load_train_act_data(nrows=ROW)
    # train_act_df = train_act_df.join(tmp_df, how='left')
    # train_df = pd.concat([train_df, train_act_df], axis=0)
    # del train_act_df, tmp_df
    # del train_act_df

    # logger.info('Train Act Data load end {}'.format(train_act_df.shape))
    # logger.info('Train Data Concat End {}'.format(train_df.shape))

    # All Data Loading
    train_index = train_df.index
    train_row = train_index.shape[0]
    test_index = test_df.index
    test_row = test_index.shape[0]
    # train_act_index = train_act_df.index
    # train_act_row = train_act_index.shape[0]

    # Target Encoding
    cat_vars = ["user_id",
                "region",
                "city",
                "parent_category_name",
                "category_name",
                "item_seq_number",
                "user_type",
                # "population",
                # "param_1",
                # "param_2",
                # "param_3",
                "image_top_1"
                ]
    '''for col in tqdm(cat_vars):
        train_tmp, test_tmp = target_encode(train_df[col],
                                            test_df[col],
                                            target=train_df['deal_probability'],
                                            min_samples_leaf=100,
                                            smoothing=10,
                                            noise_level=0.01)
        train_df = pd.concat((train_df, train_tmp), axis=1)
        test_df = pd.concat((test_df, test_tmp), axis=1)

    # train_df.to_csv("../input/data03_train.csv", index=True)
    # test_df.to_csv("../input/data04_test.csv", index=True)
    del train_tmp, test_tmp

    logger.info('Target Encoding End')'''

    train_y = train_df["deal_probability"].copy().clip(0.0, 1.0)
    train_df.drop(["deal_probability"], axis=1, inplace=True)

    df = pd.concat([train_df, test_df], axis=0)
    del train_df, test_df
    # df = pd.concat([train_df, test_df, train_act_df], axis=0)
    # del train_df, test_df, train_act_df
    gc.collect()
    logger.info('All load end {}'.format(df.shape))

    # Feature Engineering
    df["activation_weekday"] = df["activation_date"].dt.weekday
    df["price"] = np.log(df["price"] + 0.001)
    df["price"].fillna(-999, inplace=True)
    df["image_top_1"].fillna(-999, inplace=True)
    # df["population"] = np.log(df["population"] + 0.001)
    # df["population"].fillna(-999, inplace=True)

    # train_ix = df.loc[df['activation_date'] <= pd.to_datetime('2017-04-07')].index
    # valid_ix = df.loc[df['activation_date'] >= pd.to_datetime('2017-04-08')].index
    df.drop(['activation_date', 'image'], axis=1, inplace=True)
    
    # Label encode the categorical variables
    lbl = preprocessing.LabelEncoder()
    for col in tqdm(list(cat_vars)):
        df[col].fillna("Unknown")
        df[col] = lbl.fit_transform(df[col].astype(str))
    
    logger.info('Label Encode End')

    # Count Encoding
    # for col in tqdm(list(cat_vars)):
    #    df[col + '_count'] = df[col].apply(df[col].value_counts())

    # Text length Encoding
    df['param_feat'] = df.apply(lambda row: ' '.join([
                    str(row['param_1']),
                    str(row['param_2']),
                    str(row['param_3'])
                    ]), axis=1)
    df.drop(['param_1', 'param_2', 'param_3'], axis=1, inplace=True)
    
    logger.info('Param_Feat Create End')

    df['description'] = df['description'].apply(lambda x: cleanName(x))
    df['title'] = df['title'].apply(lambda x: cleanName(x))

    logger.info('CleanName End')

    text_feat = ['description', 'param_feat', 'title']
    # text_feat = ['description', 'title']
    for cols in tqdm(list(text_feat)):
        df[cols] = df[cols].astype('str')
        df[cols] = df[cols].astype('str').fillna(' ')
        df[cols] = df[cols].str.lower()
        df[cols + '_text_len'] = df[cols].apply(len)
        df[cols + '_words_len'] = df[cols].apply(lambda comment: len(comment.split()))
        df[cols + '_uni_words_len'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols + '_uni_words_len'] / df[cols + '_words_len'] * 100
        df[cols + '_words_vs_unique'].fillna(-1, inplace=True)

    logger.info('Text_Feat Create End')

    # TF-IDF
    russian_stop = set(stopwords.words('russian'))
    tfidf_param = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        # "min_df": 15,
        # "max_df": 0.3,
        "smooth_idf": False
        }

    vectorizer = FeatureUnion([
        ('description', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=21000,
            **tfidf_param,
            preprocessor=get_col('description'))),
        ('param_feat', CountVectorizer(
            ngram_range=(1, 2),
            max_features=7000,
            preprocessor=get_col('param_feat'))),
        ('title', TfidfVectorizer(
            ngram_range=(1, 2),
            # max_features=7000,
            **tfidf_param,
            preprocessor=get_col('title')))
        ])

    logger.info('Vectorization Start')
    start_vect = time.time()

    vectorizer.fit(df.loc[train_index, :].to_dict('records'))
    ready_df = vectorizer.transform(df.to_dict('records'))

    # Save Data
    # sparse.save_npz('../input/train_vec_08.npz', ready_df)
    
    # Load Data
    # start_vect = time.time()
    # ready_df = sparse.load_npz('../input/train_vec_08.npz')

    df.drop(text_feat, axis=1, inplace=True)
    logger.info('Vectorization Runtime: {} Min'.format((time.time() - start_vect) / 60))

    ready_df = ready_df[:(train_row + test_row)]
    df = df[:(train_row + test_row)]

    # Ridge Feature Processing
    ridge_train, ridge_test = ens_ridge(ready_df[:train_row],
                                        train_y, train_row,
                                        ready_df[train_row:], test_row)
    ridge_preds = np.concatenate([ridge_train, ridge_test])
    df['ridge_preds'] = ridge_preds
    del ridge_preds, ridge_train, ridge_test
    gc.collect()

    # ElasticNet Feature Processing
    '''en_train, en_test = ens_en(ready_df[:train_row],
                               train_y, train_row,
                               ready_df[train_row:], test_row)
    en_preds = np.concatenate([en_train, en_test])
    df['en_preds'] = en_preds
    del en_preds, en_train, en_test
    gc.collect()'''

    # NN Feature Processing - TFIDF
    # nn_train, nn_test = ens_nn(ready_df[:train_row],
    #                           train_y, train_row,
    #                           ready_df[train_row:], test_row)
    # NN Feature Processing - Features
    logger.debug(df.isnull().sum())
    nn_train, nn_test = ens_nn(df[:train_row], train_y, train_row,
                               df[train_row:], test_row)
    nn_preds = np.concatenate([nn_train, nn_test])
    df['nn_preds'] = nn_preds
    del nn_preds, nn_train, nn_test
    gc.collect()

    # XGB Feature Processing
    '''xgb_train, xgb_test = ens_xgb(ready_df[:train_row],
                                  train_y, train_row,
                                  ready_df[train_row:], test_row)
    xgb_preds = np.concatenate([xgb_train, xgb_test])
    df['xgb_preds'] = xgb_preds
    del xgb_preds, xgb_train, xgb_test
    gc.collect()'''

    # CatBoost Feature Processing
    '''ready_df = ready_df.toarray()
    cat_train, cat_test = ens_cat(ready_df[:train_row],
                                  train_y, train_row,
                                  ready_df[train_row:], test_row)
    cat_preds = np.concatenate([cat_train, cat_test])
    df['cat_preds'] = cat_preds
    del cat_preds, cat_train, cat_test
    gc.collect()'''


    # DataFrame => Sparce Mtrix
    tfvocab = vectorizer.get_feature_names()
    tfvocab = df.columns.tolist() + tfvocab
    logger.info('Feature Names Length:{}'.format(len(tfvocab)))

    np_train_X = df.loc[train_index, :].values
    np_test_X = df.loc[test_index, :].values

    csr_train_X = csr_matrix(np_train_X)
    csr_test_X = csr_matrix(np_test_X)
    
    train_X = hstack([csr_train_X, ready_df[:train_row]])
    test_X = hstack([csr_test_X, ready_df[train_row:]])
    
    # train_X = hstack([csr_matrix(df.loc[train_index, :].values), ready_df[:train_row]])
    # test_X = hstack([csr_matrix(df.loc[test_index, :].values), ready_df[train_row:]])
    
    del df, np_train_X, np_test_X, csr_train_X, csr_test_X,
    del ready_df, vectorizer
    gc.collect()
    logger.info('Data Preparation End & Train start')

    # Train Data Split
    train_X, valid_X, train_y, valid_y = train_test_split(train_X,
                               train_y, test_size=0.10, shuffle=False, random_state=23)

    # Trainning LightGBM
    # model = run_lgb(train_X, train_y, valid_X, valid_y)
    # model = run_lgb(train_X, train_y, valid_X, valid_y, tfvocab)
    model = run_lgb(train_X, train_y, valid_X, valid_y, tfvocab, cat_vars)

    # Trainning XGBoost
    # model = run_xgb(train_X, train_y, valid_X, valid_y)
    # model = run_xgb(train_X, train_y, valid_X, valid_y, tfvocab)

    # Test CatBoost
    # train_X = train_X.toarray()
    # valid_X = valid_X.toarray()
    # model = run_cat(train_X, train_y, valid_X, valid_y)
    
    # Test Ridge
    # model = run_ridge(train_X, train_y, valid_X, valid_y)

    # Test NN
    # model = run_nn(train_X, train_y, valid_X, valid_y)

    # with open('./models/model_xgb.pkl', mode='wb') as f:
    #  pkl.dump(model, f)

    logger.info('Train End & Test Start')

    # Predict
    # with open('./models/model_lgb_08.pkl', mode='rb') as f:
    #    model = pkl.load(f)

    # Test LightGBM
    pred_test = model.predict(test_X, num_iteration=model.best_iteration)
 
    # Test XGBoost
    # test_X = xgb.DMatrix(test_X)
    # test_X = xgb.DMatrix(test_X, feature_names=tfvocab)
    # pred_test = model.predict(test_X, ntree_limit=model.best_ntree_limit)

    # Test CatBoost
    # test_X = test_X.toarray()
    # pred_test = model.predict(test_X)

    # Test Ridge
    # pred_test = model.predict(test_X)

    # Test NN
    # test_X = test_X.tocsr()    # coo -> csr
    # pred_test = model.predict(test_X)
    
    # Making Submmit Files
    pred_test[pred_test > 1] = 1
    pred_test[pred_test < 0] = 0

    sub_df = pd.DataFrame({"item_id": test_index})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv(DIR + "submit.csv", index=False)
    # sub_df.to_csv(DIR + "train_act_pred00.csv", index=False)

    logger.info('Test End - Finish')
