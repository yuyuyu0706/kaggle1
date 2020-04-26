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
from sklearn.linear_model import Ridge
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

ROW = None
DIR = './result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
XGBFMAP = DIR + 'xgb.fmap'
TRN_PRED_FILE = DIR + 'train_act_0000_pred_lgbm.csv'
SEED = 23
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
    logger.info('rmse - Ridge Train: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def ens_nn(trn_X, trn_y, trn_rows, test_X, test_rows):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    pred_trn_X = np.zeros((trn_rows,))
    pred_test_X = np.zeros((test_rows,))
    pred_test_skf = np.empty((NFOLDS, test_rows))
    
    lr = 0.1
    bz = int(trn_X.shape[0] / 10)
    ep = 50
    op = ks.optimizers.Adam(lr=lr)

    model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=op)

    for i, (trn_i, test_i) in tqdm(list(enumerate(kf.split(trn_X)))):
        x_trn = trn_X[trn_i]
        y_trn = trn_y[trn_i]
        x_test = trn_X[test_i]

        for j in range(ep):
            model.fit(x=x_trn, y=y_trn, batch_size=bz, epochs=1, verbose=0)
            pred_trn = model.predict(x_trn, batch_size=bz)[:, 0]
            rmse_trn = rmse(y_trn, pred_trn)
            logger.debug('epochs {0}: rmse - Train:{1:.6f}'.format(j+1, rmse_trn))

        pred_trn_X[test_i] = model.predict(x_test, batch_size=bz)[:, 0]
        pred_test_skf[i, :] = model.predict(test_X, batch_size=bz)[:, 0]

    pred_test_X[:] = pred_test_skf.mean(axis=0)

    rmse_trn = rmse(trn_y, pred_trn_X)
    logger.info('rmse - NN Train: {}'.format(rmse_trn))

    return pred_trn_X.reshape(-1, 1), pred_test_X.reshape(-1, 1)

def run_nn(trn_X, trn_y, val_X, val_y):
    lr = 0.1
    bz = int(trn_X.shape[0] / 10)
    ep = 50
    op = ks.optimizers.Adam(lr=lr)
    # op = ks.optimizers.SGD(lr=0.001, momentum=0.9)
    # with tf.Session(graph=tf.Graph(), config=config) as sess:
    early = ks.callbacks.EarlyStopping(patience=5, mode='min')
    model_in = ks.Input(shape=(trn_X.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=op)
    model.fit(x=trn_X, y=trn_y, batch_size=bz, epochs=ep, callbacks=[early], verbose=1)
    '''
    for i in range(ep):
        model.fit(x=trn_X, y=trn_y, batch_size=bz, epochs=1, verbose=1)
        pred_trn = model.predict(trn_X, batch_size=bz)[:, 0]
        pred_val = model.predict(val_X, batch_size=bz)[:, 0]
        rmse_trn = rmse(trn_y, pred_trn)
        rmse_val = rmse(val_y, pred_val)
        logger.info('epochs {0}: rmse - Train:{1:.4f} Valid:{2:.4f}'.format(i+1, rmse_trn, rmse_val))
    '''
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

    logger.info('split.train: {}'.format(trn_X.shape))
    logger.info('split.valid: {}'.format(val_X.shape))

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
    logger.info('test load end {}'.format(test_df.shape))

    # test_df = load_period_train_data(nrows=ROW)
    # logger.info('period train load end {}'.format(test_df.shape))

    # pr_test_df = load_period_test_data(nrows=ROW)
    # logger.info('period test load end {}'.format(pr_test_df.shape))

    # test_df = load_train_act_data(nrows=ROW)
    # tmp_df = pd.read_csv(TRN_PRED_FILE, index_col=['item_id'])
    # trn_act_df = load_train_act_data(nrows=ROW)
    # trn_act_df = trn_act_df.join(tmp_df, how='left')
    # train_df = pd.concat([train_df, trn_act_df], axis=0)
    # del trn_act_df, tmp_df

    # logger.info('Train Act Data load end {}'.format(train_act_df.shape))
    # logger.info('Train Data Concat End {}'.format(train_df.shape))


    # All Data Loading
    train_index = train_df.index
    train_row = train_index.shape[0]
    test_index = test_df.index
    test_row = test_index.shape[0]

    train_y = train_df["deal_probability"].copy().clip(0.0, 1.0)
    train_df.drop(["deal_probability"], axis=1, inplace=True)

    df = pd.concat([train_df, test_df], axis=0)
    del train_df, test_df
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
    cat_vars = ["user_id",
                "region",
                "city",
                "parent_category_name",
                "category_name",
                "item_seq_number",
                "user_type",
                # "population",
                "param_1",
                "param_2",
                "param_3",
                "image_top_1"

                ]
    lbl = preprocessing.LabelEncoder()
    for col in tqdm(list(cat_vars)):
        df[col].fillna("Unknown")
        df[col] = lbl.fit_transform(df[col].astype(str))

    logger.info('Data Preparation End')

    '''
    df['param_feat'] = df.apply(lambda row: ' '.join([
                    str(row['param_1']),
                    str(row['param_2']),
                    str(row['param_3'])
                    ]), axis=1)
    df.drop(['param_1', 'param_2', 'param_3'], axis=1, inplace=True)
    '''

    df['description'] = df['description'].apply(lambda x: cleanName(x))
    df['title'] = df['title'].apply(lambda x: cleanName(x))

    # text_feat = ['description', 'param_feat', 'title']
    text_feat = ['description', 'title']
    for cols in tqdm(list(text_feat)):
        df[cols] = df[cols].astype('str')
        df[cols] = df[cols].astype('str').fillna(' ')
        df[cols] = df[cols].str.lower()
        df[cols + '_text_len'] = df[cols].apply(len)
        df[cols + '_words_len'] = df[cols].apply(lambda comment: len(comment.split()))
        df[cols + '_uni_words_len'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols + '_uni_words_len'] / df[cols + '_words_len'] * 100

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
        # ('param_feat', CountVectorizer(
        #    ngram_range=(1, 2),
        #    max_features=7000,
        #    preprocessor=get_col('param_feat'))),
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

    # Ridge Feature Processing
    ridge_train, ridge_test = ens_ridge(ready_df[:train_row],
                                        train_y, train_row,
                                        ready_df[train_row:], test_row)
    ridge_preds = np.concatenate([ridge_train, ridge_test])
    df['ridge_preds'] = ridge_preds

    # NN Feature Processing
    # nn_train, nn_test = ens_nn(ready_df[:train_row],
    #                           train_y, train_row,
    #                           ready_df[train_row:], test_row)
    # nn_preds = np.concatenate([nn_train, nn_test])
    # df['nn_preds'] = nn_preds

    # Feature Stack
    tfvocab = vectorizer.get_feature_names()
    tfvocab = df.columns.tolist() + tfvocab
    logger.info('Feature Names Length:{}'.format(len(tfvocab)))

    csr_train_X = csr_matrix(df.loc[train_index, :].values)
    csr_test_X = csr_matrix(df.loc[test_index, :].values)
    train_X = hstack([csr_train_X, ready_df[:train_row]])
    test_X = hstack([csr_test_X, ready_df[train_row:]])
    # train_X = hstack([csr_matrix(df.loc[train_index, :].values), ready_df[:train_row]])
    # test_X = hstack([csr_matrix(df.loc[test_index, :].values), ready_df[train_row:]])
    
    del df
    gc.collect()

    # Train Data Split
    train_X, valid_X, train_y, valid_y = train_test_split(train_X,
                             train_y, test_size=0.10, shuffle=False, random_state=23)

    # del ready_df, vectorizer, ridge_preds, nn_preds 
    del ready_df, vectorizer, ridge_preds
    logger.info('Data Preparation End & Train start')

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
