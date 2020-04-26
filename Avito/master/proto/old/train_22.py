import time
import gc
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle as pkl
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

# Keras
import keras as ks
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, GRU
from keras.layers import GlobalMaxPool1D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

# TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

ROW = None
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

def do_count(df, group_cols, agg_name):
    gp = df[group_cols][group_cols].groupby(group_cols).size()
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def do_count_uniq(df, group_cols, counted, agg_name):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique()
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def do_count_mean(df, group_cols, counted, agg_name):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean()
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def do_count_var(df, group_cols, counted, agg_name):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var()
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def do_count_var_ddof(df, group_cols, counted, agg_name):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var(ddof=False)
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def do_count_std(df, group_cols, counted, agg_name):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].std()
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def do_count_std_ddof(df, group_cols, counted, agg_name):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].std(ddof=False)
    gp = gp.rename(agg_name)
    gp = gp.to_frame()
    df = pd.merge(df, gp, left_on=group_cols, right_index=True, how='left')
    del gp
    return df

def tknzr_fit(col, df_trn, df_test):
    tknzr = Tokenizer(filters='', lower=False, split='뷁', oov_token='oov')
    tknzr.fit_on_texts(df_trn[col])
    tknzr_trn = np.array(tknzr.texts_to_sequences(df_trn[col]))
    tknzr_test = np.array(tknzr.texts_to_sequences(df_test[col]))

    return tknzr_trn, tknzr_test, tknzr

def tknzr_desc_fit(col, df_trn, df_test):
    tknzr = Tokenizer(num_words=100000, lower=False)
    tknzr.fit_on_texts(df_trn[col].values)
    trn_seq = tknzr.texts_to_sequences(df_trn[col].values)
    test_seq = tknzr.texts_to_sequences(df_test[col].values)
    trn_pad = pad_sequences(trn_seq, maxlen=75)
    test_pad = pad_sequences(test_seq, maxlen=75)

    return trn_pad, test_pad, tknzr

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
        # x_trn = trn_X.iloc[trn_i]
        # y_trn = trn_y.iloc[trn_i]
        # x_test = trn_X.iloc[test_i]

        model.fit(x_trn, y_trn)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - ElasticNet Train: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def make_model_nn(trn_X, test_X):
    # ks.backend.clear_session()
    ## Tarknizer
    '''tr_reg, te_reg = (trn_X['region'], test_X['region'])
    tr_city, te_city = (trn_X['city'], test_X['city'])
    tr_pcn, te_pcn = (trn_X['parent_category_name'], test_X['parent_category_name'])
    tr_cn, te_cn = (trn_X['category_name'], test_X['category_name'])
    tr_ut, te_ut = (trn_X['user_type'], test_X['user_type'])
    tr_pf, te_pf, tknzr_pf = tknzr_fit('param_feat', trn_X, test_X)
    # tr_desc, te_desc, tknzr_desc = tknzr_fit('discription', trn_X, test_X)'''
    
    ## categorical
    len_reg = len(trn_X['region'])
    len_city = len(trn_X['city']) +1
    len_pcn = len(trn_X['parent_category_name'])
    len_cn = len(trn_X['category_name'])
    len_ut = len(trn_X['user_type'])
    len_pf = len(trn_X['param_feat'])+1
    # len_week = 7
    # len_imgt1 = int(df_x_train['image_top_1'].max())+1

    #text
    len_desc = 75

    ## continuous
    len_price = 1
    len_itemseq = 1

    ## categorical
    emb_reg = 8
    emb_city = 16
    emb_pcn = 4
    emb_cn = 8
    emb_ut = 2
    emb_pf = 8
    # emb_week = 4
    # emb_imgt1 = 16

    #continuous
    emb_price = 16
    emb_itemseq = 16

    #text
    emb_desc = 100

    ## Layer - Categorical
    inp_reg = Input(shape=(1, ), name='inp_region')
    emb_reg = Embedding(len_reg, emb_reg, name='emb_region')(inp_reg)
    inp_city = Input(shape=(1, ), name='inp_city')
    emb_city = Embedding(len_city, emb_city, name='emb_city')(inp_city)
    inp_pcn = Input(shape=(1, ), name='inp_parent_category_name')
    emb_pcn = Embedding(len_pcn, emb_pcn, name='emb_parent_category_name')(inp_pcn)
    inp_cn = Input(shape=(1, ), name='inp_category_name')
    emb_cn = Embedding(len_cn, emb_cn, name='emb_category_name')(inp_cn)
    inp_ut = Input(shape=(1, ), name='inp_user_type')
    emb_ut = Embedding(len_ut, emb_ut, name='emb_user_type')(inp_ut)
    inp_pf = Input(shape=(1, ), name='inp_param_feat')
    emb_pf = Embedding(len_pf, emb_pf, name='emb_param_feat')(inp_pf)
    
    con_cat = concatenate([emb_reg, emb_city, emb_pcn, emb_cn, emb_ut, emb_pf], axis=-1, name='concat_vars')
    # con_cat = concatenate([emb_reg, emb_city, emb_pcn, emb_cn, emb_ut], axis=-1, name='concat_vars')
    
    con_cat = GlobalMaxPool1D()(con_cat)

    out_cat = Dense(200, activation='relu')(con_cat)
    out_cat = Dense(50, activation='relu')(out_cat)

    ### text
    # inp_desc = Input(shape=(len_desc, ), name='inp_description')
    # emb_desc = Embedding(len_desc, emb_desc, name='emb_description')(inp_desc)
    # out_desc = GRU(40, return_sequences=False)(emb_desc)
    # out = concatenate([out_cat, out_desc], axis=-1)

    ### Output
    out = Dense(1, activation='sigmoid', name='output')(out_cat)
    # out = Dense(1, activation='sigmoid', name='output')(out)

    # model = Model(inputs = [inp_reg, inp_city, inp_pcn, inp_cn, inp_ut], outputs = out)
    model = Model(inputs = [inp_reg, inp_city, inp_pcn, inp_cn, inp_ut, inp_pf], outputs = out)
    # model = Model(inputs = [inp_reg, inp_city, inp_pcn, inp_cn, inp_ut, inp_pf, inp_desc], outputs = out)

    # df to np
    # X_trn = np.array([tr_reg, tr_city, tr_pcn, tr_cn, tr_ut])
    # X_test = np.array([te_reg, te_city, te_pcn, te_cn, te_ut])
    # X_trn = [x for x in X_trn]
    # X_test = [x for x in X_test]
    
    # return model, X_trn, X_test
    return model

def ens_nn(trn_X, trn_y, trn_rows, test_X, test_rows):

    # trn_X['param_feat'], test_X['param_feat'], tknzr_pf = tknzr_fit('param_feat', trn_X, test_X)
    # trn_X['description'], test_X['description'], tknzr_pf = tknzr_desc_fit('description', trn_X, test_X)

    trn_X = trn_X.ix[:, [1, 2, 3, 4, 8, 11, 5]]
    test_X = test_X.ix[:, [1, 2, 3, 4, 8, 11, 5]]

    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    lr = 0.005
    bz = 100000
    ep = 500
    op = Adam(lr=lr)
    # early = EarlyStopping(monitor='val_loss', patience=500, mode='min')

    logger.info('NN Train Shape: {}'.format(trn_X.shape))
    logger.info('NN Test Shape : {}'.format(test_X.shape))

    '''# model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=True)
    # model_in = Input(shape=(trn_X.shape[1],), dtype='float32', sparse=False)
    # out = ks.layers.Dense(192, activation='relu')(model_in)
    # out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dense(64, activation='relu')(out)
    out = Dense(16, activation='relu')(model_in)
    out = Dense(8, activation='relu')(out)
    out = Dense(8, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(model_in, out)'''

    model = make_model_nn(trn_X, test_X)
    model.compile(loss='mean_squared_error', optimizer=op)
    
    '''tr_reg, ts_reg = ((trn_X['region']), (test_X['region']))
    tr_city, ts_city = (np.array(trn_X['city']), np.array(test_X['city']))
    tr_pcn, ts_pcn = (np.array(trn_X['parent_category_name']), np.array(test_X['parent_category_name']))
    tr_cn, ts_cn = (np.array(trn_X['category_name']), np.array(test_X['category_name']))
    tr_ut, ts_ut = (np.array(trn_X['user_type']), np.array(test_X['user_type']))
    tr_pf, ts_pf, tknzr_pf = tknzr_fit('param_feat', trn_X, test_X)'''
    trn_X['param_feat'], test_X['param_feat'], tknzr_pf = tknzr_fit('param_feat', trn_X, test_X)
    
    # trn_X = np.array([tr_reg, tr_city, tr_pcn, tr_cn, tr_ut, tr_pf[:,0]])
    # test_X = np.array([ts_reg, ts_city, ts_pcn, tr_cn, tr_ut, tr_pf[:,0]])

    trn_X = np.array(trn_X)
    test_X = np.array(test_X)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
    # for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X.index)))):
        x_trn = trn_X[trn_i]
        y_trn = trn_y[trn_i]
        # x_trn = trn_X.iloc[trn_i]
        # y_trn = trn_y.iloc[trn_i]
        x_trn, x_val, y_trn, y_val = train_test_split(
                                        x_trn, y_trn, test_size=0.10,
                                        shuffle=False, random_state=23)
        x_test = trn_X[test_i]
        # x_test = trn_X.iloc[test_i]
        
        '''x_trn = np.array(x_trn)
        y_trn = np.array(y_trn)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        x_test = np.array(x_test)
        test_X = np.array(test_X)'''

        model.fit(x=[x_trn[:,0], x_trn[:,1], x_trn[:,2], x_trn[:,3], x_trn[:,4], x_trn[:,5]], y=y_trn,
        # model.fit(x=[x_trn[:,[0,1,2,3,4,5]]], y=y_trn,
                  validation_data=([x_val[:,0], x_val[:,1], x_val[:,2], x_val[:,3], x_val[:,4], x_val[:,5]],y_val),
                  batch_size=bz, epochs=ep, verbose=1)
                 # batch_size=bz, epochs=ep, callbacks=[early], verbose=1)
        pred_trn_X[test_i] = model.predict([x_test[:,0], x_test[:,1], x_test[:,2], x_test[:,3], x_test[:,4], x_test[:,5]],
                                            batch_size=bz)[:, 0]
        pred_test_skf[i, :] = model.predict([test_X[:,0], test_X[:,1], test_X[:,2], test_X[:,3], test_X[:,4], test_X[:,5]],
                                            batch_size=bz)[:, 0]

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - NN Train: {}, lr: {}, bz: {}, ep: {}'.format(rmse_trn, lr, bz, ep))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def ens_cat(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    cat_params = {
        "iterations": 1000,
        "learning_rate": 0.08,
        "depth": 10,
        "eval_metric": 'RMSE',
        "metric_period": 50,
        "calc_feature_importance": True
        }
    
    model = CatBoostRegressor(**cat_params)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X.iloc[trn_i]
        y_trn = trn_y.iloc[trn_i]
        x_test = trn_X.iloc[test_i]

        model.fit(x_trn, y_trn, use_best_model=True)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - CatBoost Train: {}, lr: {}, dp: {}'.format(rmse_trn,
                            cat_params["learning_rate"], cat_params["depth"]))

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
        "num_estimators": 100
        }

    model = xgb.XGBRegressor(**xg_params)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X.iloc[trn_i]
        y_trn = trn_y.iloc[trn_i]
        x_test = trn_X.iloc[test_i]

        model.fit(x_trn, y_trn)

        pred_trn_X[test_i] = model.predict(x_test)
        pred_test_skf[i, :] = model.predict(test_X)

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('RMSE - XGBoost Train: {}, lr: {}, dp: {}'.format(rmse_trn,
                            xg_params["learning_rate"], xg_params["max_depth"]))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def run_nn(trn_X, trn_y, val_X, val_y):
    lr = 0.1
    bz = int(trn_X.shape[0] / 10)
    ep = 50
    op = ks.optimizers.Adam(lr=lr)
    # op = ks.optimizers.SGD(lr=0.001, momentum=0.9)
    # with tf.Session(graph=tf.Graph(), config=config) as sess:
    # early = ks.callbacks.EarlyStopping(monitor='loss', patience=0, mode='min')
    
    model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    # trn_X = trn_X.ix[:, [1, 2, 3, 4, 8, 11, 5]]
    # val_X = val_X.ix[:, [1, 2, 3, 4, 8, 11, 5]]
    # model = make_model_nn(trn_X, val_X)

    model.compile(loss='mean_squared_error', optimizer=op)
    model.fit(x=trn_X, y=trn_y, validation_data=(val_X, val_y),
              batch_size=bz, epochs=ep, verbose=1)
    #          batch_size=bz, epochs=ep, callbacks=[early], verbose=1)
    
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
        "num_leaves": 128,      # [256]
        "learning_rate": 0.07,   # [0.018]
        "feature_fraction": 0.7, # [0.5]
        "bagging_freq": 5,
        "bagging_fraction": 0.7, # [0.75]
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
    for i in range(500):
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
    
    # train_df["price"].fillna(-999, inplace=True)
    # train_df = train_df[train_df['price'] < 1000000000]
    # train_df.reset_index(drop=True)
   
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

    # Negative Down Sampling
    # train_df = train_df[train_df['deal_probability'] > 0]
    # train_df.reset_index(inplace=True, drop=True)
    # logger.info('Train Data Negative Down {}'.format(train_df.shape))
   
    # All Data Loading
    train_index = train_df.index
    train_row = train_index.shape[0]
    test_index = test_df.index
    test_row = test_index.shape[0]
    # train_act_index = train_act_df.index
    # train_act_row = train_act_index.shape[0]

    # Target Encoding
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
    df["week"] = df["activation_date"].dt.weekday
    df["price"] = np.log(df["price"] + 0.001)
    df["price"].fillna(-999, inplace=True)
    df["image_top_1"].fillna(-999, inplace=True)
    # df["population"] = np.log(df["population"] + 0.001)
    # df["population"].fillna(-999, inplace=True)

    # train_ix = df.loc[df['activation_date'] <= pd.to_datetime('2017-04-07')].index
    # valid_ix = df.loc[df['activation_date'] >= pd.to_datetime('2017-04-08')].index
    df.drop(['activation_date', 'image'], axis=1, inplace=True)
    
    # categorical variables
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
                "image_top_1",
                "week"
                ]
    # Label encode the categorical variables
    lbl = preprocessing.LabelEncoder()
    for col in tqdm(list(cat_vars)):
        df[col].fillna("Unknown")
        df[col] = lbl.fit_transform(df[col].astype(str))
    
    logger.info('Label Encode End')

    # Count Encoding
    for col in tqdm(list(cat_vars)):
        df = do_count(df, [col], col + '_count')
    
    # Count Encoding - multi
    df = do_count(df, ['user_id', 'region'], 'uid_region_count')
    df = do_count(df, ['user_id', 'city'], 'uid_city_count')
    df = do_count(df, ['user_id', 'region', 'city'], 'uid_region_city_count')
    df = do_count(df, ['user_id', 'region', 'city', 'user_type'], 'uid_region_city_ut_count')
    df = do_count(df, ['user_id', 'week'], 'uid_week_count')
    df = do_count(df, ['user_id', 'parent_category_name', 'category_name'], 'uid_pcn_cn_count')
    df = do_count(df, ['user_id', 'item_seq_number'], 'uid_isn_count')
    df = do_count(df, ['user_id', 'image_top_1'], 'uid_it1_count')
    
    # Count Encoding - uniq
    df = do_count_uniq(df, ['region'], 'user_id', 'uid_by_region_count_uniq')
    df = do_count_uniq(df, ['city'], 'user_id', 'uid_by_city_count_uniq')
    df = do_count_uniq(df, ['week'], 'user_id', 'uid_by_week_count_uniq')
    df = do_count_uniq(df, ['parent_category_name', 'category_name'], 'user_id', 'uid_by_pcn_cn_count_uniq')
    df = do_count_uniq(df, ['item_seq_number', 'image_top_1'], 'user_id', 'uid_by_isn_it1_count_uniq')
    df = do_count_uniq(df, ['image_top_1'], 'user_id', 'uid_by_it1_count_uniq')
    df = do_count_uniq(df, ['image_top_1'], 'region', 'region_by_it1_count_uniq')
    df = do_count_uniq(df, ['image_top_1'], 'city', 'city_by_it1_count_uniq')
    df = do_count_uniq(df, ['image_top_1'], 'week', 'week_by_it1_count_uniq')
    df = do_count_uniq(df, ['item_seq_number'], 'user_id', 'uid_by_isn_count_uniq')
    df = do_count_uniq(df, ['item_seq_number'], 'region', 'region_by_isn_count_uniq')
    df = do_count_uniq(df, ['item_seq_number'], 'city', 'city_by_isn_count_uniq')
    df = do_count_uniq(df, ['item_seq_number'], 'week', 'week_by_isn_count_uniq')
    df = do_count_uniq(df, ['item_seq_number'], 'image_top_1', 'it1_by_isn_count_uniq')
    
    # Count Encoding - mean
    df = do_count_mean(df, ['price'], 'user_id', 'uid_by_price_mean')
    df = do_count_mean(df, ['price'], 'uid_by_region_count_uniq', 'uid_by_price_loc_mean')
    df = do_count_mean(df, ['price'], 'uid_by_pcn_cn_count_uniq', 'uid_by_price_cat_mean')
    df = do_count_mean(df, ['price'], 'week', 'week_by_price_mean')
    df = do_count_mean(df, ['price'], 'image_top_1', 'it1_by_price_mean')
    df = do_count_mean(df, ['price'], 'item_seq_number', 'isn_by_price_mean')
    
    # Count Encoding - variance
    df = do_count_var(df, ['price'], 'user_id', 'uid_by_price_var')
    df = do_count_var(df, ['price'], 'uid_by_region_count_uniq', 'uid_by_price_loc_var')
    df = do_count_var(df, ['price'], 'uid_by_pcn_cn_count_uniq', 'uid_by_price_cat_var')
    df = do_count_var(df, ['price'], 'week', 'week_by_price_var')
    df = do_count_var(df, ['price'], 'image_top_1', 'it1_by_price_var')
    df = do_count_var(df, ['price'], 'item_seq_number', 'isn_by_price_var')

    # Count Encoding - variance-ddof
    df = do_count_var_ddof(df, ['price'], 'user_id', 'uid_by_price_var_ddof')
    df = do_count_var_ddof(df, ['price'], 'uid_by_region_count_uniq', 'uid_by_price_loc_var_ddof')
    df = do_count_var_ddof(df, ['price'], 'uid_by_pcn_cn_count_uniq', 'uid_by_price_cat_var_ddof')
    df = do_count_var_ddof(df, ['price'], 'week', 'week_by_price_var_ddof')
    df = do_count_var_ddof(df, ['price'], 'image_top_1', 'it1_by_price_var_ddof')
    df = do_count_var_ddof(df, ['price'], 'item_seq_number', 'isn_by_price_var_ddof')

    # Count Encoding - standard
    df = do_count_std(df, ['price'], 'user_id', 'uid_by_price_std')
    df = do_count_std(df, ['price'], 'uid_by_region_count_uniq', 'uid_by_price_loc_std')
    df = do_count_std(df, ['price'], 'uid_by_pcn_cn_count_uniq', 'uid_by_price_cat_std')
    df = do_count_std(df, ['price'], 'week', 'week_by_price_std')
    df = do_count_std(df, ['price'], 'image_top_1', 'it1_by_price_std')
    
    # Count Encoding - standard-ddof
    df = do_count_std_ddof(df, ['price'], 'user_id', 'uid_by_price_std_ddof')
    df = do_count_std_ddof(df, ['price'], 'uid_by_region_count_uniq', 'uid_by_price_loc_std_ddof')
    df = do_count_std_ddof(df, ['price'], 'uid_by_pcn_cn_count_uniq', 'uid_by_price_cat_std_ddof')
    df = do_count_std_ddof(df, ['price'], 'week', 'week_by_price_std_ddof')
    df = do_count_std_ddof(df, ['price'], 'image_top_1', 'it1_by_price_std_ddof')
    
    logger.info('Count Encode End')

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
    name = {'_text_len': '_tl_by_price',
            '_words_len': '_wl_by_price',
            '_words_uniq_len': '_wul_by_price',
            '_words_vs_uniq': '_wvu_by_price',
            '_num_letters': '_nl_by_price'
            }
    for cols in tqdm(list(text_feat)):
        df[cols] = df[cols].astype('str')
        df[cols] = df[cols].astype('str').fillna(' ')
        df[cols] = df[cols].str.lower()
        df[cols + '_text_len'] = df[cols].apply(len)
        df[cols + '_words_len'] = df[cols].apply(lambda comment: len(comment.split()))
        df[cols + '_words_uniq_len'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_uniq'] = df[cols + '_words_uniq_len'] / df[cols + '_words_len'] * 100
        df[cols + '_words_vs_uniq'].fillna(-1, inplace=True)
        df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment))
        df[cols + '_num_alphabets'] = df[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]')))
        df[cols + '_num_alphanumeric'] = df[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]')))
        
        for cols2 in tqdm(list(name.items())):
            df = do_count_mean(df, ['price'], cols + cols2[0], cols + cols2[1] + '_mean')
            df = do_count_var(df, ['price'], cols + cols2[0], cols + cols2[1] + '_var')
            df = do_count_var_ddof(df, ['price'], cols + cols2[0], cols + cols2[1] + '_var_ddof')
            df = do_count_std(df, ['price'], cols + cols2[0], cols + cols2[1] + '_std')
            df = do_count_std_ddof(df, ['price'], cols + cols2[0], cols + cols2[1] + '_std_ddof')
            df[cols + cols2[1] + '_mean'].fillna(-1, inplace=True)
            df[cols + cols2[1] + '_var'].fillna(-1, inplace=True)
            df[cols + cols2[1] + '_var_ddof'].fillna(-1, inplace=True)
            df[cols + cols2[1] + '_std'].fillna(-1, inplace=True)
            df[cols + cols2[1] + '_std_ddof'].fillna(-1, inplace=True)

    # Extra Feature Engineering
    df['title_desc_text_len_ratio'] = df['title_text_len']/df['description_text_len']
    df['title_desc_words_len_ratio'] = df['title_words_len']/df['description_words_len']
    df['title_desc_words_uniq_len_ratio'] = df['title_words_uniq_len']/df['description_words_uniq_len']
    df['title_desc_wvu_ratio'] = df['title_words_vs_uniq']/df['description_words_vs_uniq']
    df['title_desc_num_let_len_ratio'] = df['title_num_letters']/df['description_num_letters']
    
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

    # df.drop(text_feat, axis=1, inplace=True)
    # df.drop(['description', 'title'], axis=1, inplace=True)
    df.drop(['title'], axis=1, inplace=True)
    logger.info('Vectorization Runtime: {} Min'.format((time.time() - start_vect) / 60))

    ready_df = ready_df[:(train_row + test_row)]
    df = df[:(train_row + test_row)]

    # ens_X = df[:train_row]
    # ens_y = train_y
    # nn_X, ens_X, nn_y, ens_y = train_test_split(ens_X, ens_y, test_size=0.30, shuffle=False, random_state=23)
    # xgb_X, cat_X, xgb_y, cat_y = train_test_split(ens_X, ens_y, test_size=0.30, shuffle=False, random_state=23)

    # Ridge Feature Processing - TFIDF
    # ridge_train, ridge_test = ens_ridge(ready_df[:train_row],
    #                                    train_y, train_row,
    #                                    ready_df[train_row:], test_row)
    # ridge_preds = np.concatenate([ridge_train, ridge_test])
    # df['ridge_preds'] = ridge_preds
    # del ridge_preds, ridge_train, ridge_test
    # gc.collect()

    # ElasticNet Feature Processing - TFIDF
    # en_train, en_test = ens_en(ready_df[:train_row],
    #                            train_y, train_row,
    #                            ready_df[train_row:], test_row)
    # ElasticNet Feature Processing
    # en_train, en_test = ens_en(df[:train_row], train_y, train_row,
    #                            df[train_row:], test_row)
    # en_preds = np.concatenate([en_train, en_test])
    # df['en_preds'] = en_preds
    # del en_preds, en_train, en_test
    # gc.collect()

    # NN Feature Processing - TFIDF
    # nn_train, nn_test = ens_nn(ready_df[:train_row],
    #                           train_y, train_row,
    #                           ready_df[train_row:], test_row)
    # NN Feature Processing - Features
    '''nn_train, nn_test = ens_nn(df[:train_row], train_y, train_row,
                               df[train_row:], test_row)
    nn_preds = np.concatenate([nn_train, nn_test])
    del nn_train, nn_test
    gc.collect()'''

    # df.drop(text_feat, axis=1, inplace=True)
    df.drop(['param_feat'], axis=1, inplace=True)
    df.drop(['description'], axis=1, inplace=True)
    
    # XGB Feature Processing - TFIDF
    # xgb_train, xgb_test = ens_xgb(ready_df[:train_row],
    #                               train_y, train_row,
    #                               ready_df[train_row:], test_row)
    # XGB Feature Processing - Features
    '''xgb_train, xgb_test = ens_xgb(df[:train_row], train_y, train_row,
                                  df[train_row:], test_row)
    xgb_preds = np.concatenate([xgb_train, xgb_test])
    del xgb_train, xgb_test
    gc.collect()'''

    # CatBoost Feature Processing - TFIDF
    # ready_df = ready_df.toarray()
    # cat_train, cat_test = ens_cat(ready_df[:train_row],
    #                               train_y, train_row,
    #                               ready_df[train_row:], test_row)
    # CatBoost Feature Processing - TFIDF
    '''cat_train, cat_test = ens_cat(df[:train_row], train_y, train_row,
                                  df[train_row:], test_row)
    cat_preds = np.concatenate([cat_train, cat_test])
    del cat_train, cat_test
    gc.collect()'''

    # Stack_predict
    # df['nn_preds'] = nn_preds
    # df['xgb_preds'] = xgb_preds
    # df['cat_preds'] = cat_preds
    # del nn_preds, xgb_preds, cat_preds
    # del xgb_preds
    # gc.collect()
    
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
    
    del train_X, train_y, valid_X, valid_y, tfvocab, cat_vars
    gc.collect()
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
