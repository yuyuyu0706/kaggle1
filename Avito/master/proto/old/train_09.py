import time
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle as pkl
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split
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

ROW = 100
DIR = './result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
XGBFMAP = DIR + 'xgb.fmap'

def get_col(col_name):
    return lambda x: x[col_name]

def rmse(label_y, pred_y):
    return np.sqrt(mean_squared_error(label_y, pred_y))

def create_feats_map(features):
    with open(XGBFMAP, 'w') as f:
        i = 0
        for feat in features:
            f.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

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
        "max_depth": 15,
        # "num_leaves": 128,      # [32, 48, 64, 128]
        "learning_rate": 0.07,  # [0.05, 0.07, 0.1, 0.2]
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
    for i in range(15):
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

    train_df = load_train_data(nrows=ROW)
    logger.info('Train Data load end {}'.format(train_df.shape))

    test_df = load_test_data(nrows=ROW)
    logger.info('test load end {}'.format(test_df.shape))

    # test_df = load_period_train_data(nrows=ROW)
    # logger.info('period train load end {}'.format(test_df.shape))

    # pr_test_df = load_period_test_data(nrows=ROW)
    # logger.info('period test load end {}'.format(pr_test_df.shape))

    # test_df = load_train_act_data(nrows=ROW)
    # logger.info('Train Act Data load end {}'.format(test_df.shape))

    # All Data Loading
    train_index = train_df.index
    test_index = test_df.index
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

    # train_ix = df.loc[df['activation_date'] <= pd.to_datetime('2017-04-07')].index
    # valid_ix = df.loc[df['activation_date'] >= pd.to_datetime('2017-04-08')].index
    df.drop(['activation_date', 'image'], axis=1, inplace=True)
    logger.info('Data Preparation End')
    
    # Label encode the categorical variables
    cat_vars = ["user_id",
                "region",
                "city",
                "parent_category_name",
                "category_name",
                "item_seq_number",
                "user_type",
                "image_top_1"
                ]
    lbl = preprocessing.LabelEncoder()
    for col in tqdm(list(cat_vars)):
        df[col] = lbl.fit_transform(df[col].astype(str))

    df['param_feat'] = df.apply(lambda row: ' '.join([
                    str(row['param_1']),
                    str(row['param_2']),
                    str(row['param_3'])
                    ]), axis=1)
    df.drop(['param_1', 'param_2', 'param_3'], axis=1, inplace=True)

    text_feat = ['description', 'param_feat', 'title']
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
        # "min_df": 5,
        # "max_df": .9,
        "smooth_idf": False
        }

    vectorizer = FeatureUnion([
        ('description', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100,
            **tfidf_param,
            preprocessor=get_col('description'))),
        ('param_feat', CountVectorizer(
            ngram_range=(1, 2),
            max_features=100,
            preprocessor=get_col('param_feat'))),
        ('title', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100,
            **tfidf_param,
            preprocessor=get_col('title')))
        ])

    start_vect = time.time()
    vectorizer.fit(df.loc[train_index, :].to_dict('records'))
    ready_df = vectorizer.transform(df.to_dict('records'))
    tfvocab = vectorizer.get_feature_names()
    logger.info('Vectorization Runtime: {} Min'.format((time.time() - start_vect) / 60))

    df.drop(text_feat, axis=1, inplace=True)

    # print(train_index.shape[0])
    # print(df.loc[test_index, :].shape)
    # print(test_index.shape[0])

    train_X = hstack([csr_matrix(df.loc[train_index, :].values), ready_df[0:train_index.shape[0]]])
    test_X = hstack([csr_matrix(df.loc[test_index, :].values), ready_df[train_index.shape[0]:]])
    
    tfvocab = df.columns.tolist() + tfvocab
    create_feats_map(list(df.columns[2:]))
    logger.info('Feature Names Length:{}'.format(len(tfvocab)))
    
    del df
    gc.collect()

    # Train Data Split
    train_X, valid_X, train_y, valid_y = train_test_split(train_X,
                             train_y, test_size=0.10, shuffle=False, random_state=23)

    logger.info('Data Preparation End & Train start')

    # Trainning LightGBM
    # model = run_lgb(train_X, train_y, valid_X, valid_y)
    # model = run_lgb(train_X, train_y, valid_X, valid_y, tfvocab, cat_vars)

    # Trainning XGBoost
    # model = run_xgb(train_X, train_y, valid_X, valid_y)
    # model = run_xgb(train_X, train_y, valid_X, valid_y, tfvocab)

    # Test CatBoost
    train_X = train_X.toarray()
    valid_X = valid_X.toarray()
    model = run_cat(train_X, train_y, valid_X, valid_y)
    
    # with open('./model_lgb.pkl', mode='wb') as f:
    #   pkl.dump(model, f)

    logger.info('Train End')
    logger.info('')
    logger.info('Test Start')

    # Predict
    # with open('./model_lgb_08.pkl', mode='rb') as f:
    #    model = pkl.load(f)

    # Test LightGBM
    # pred_test = model.predict(test_X, num_iteration=model.best_iteration)
 
    # Test XGBoost
    # test_X = xgb.DMatrix(test_X)
    # test_X = xgb.DMatrix(test_X, feature_names=tfvocab)

    # pred_test = model.predict(test_X, ntree_limit=model.best_ntree_limit)

    # Test CatBoost
    test_X = test_X.toarray()
    pred_test = model.predict(test_X)

    # Making Submmit Files
    pred_test[pred_test > 1] = 1
    pred_test[pred_test < 0] = 0

    sub_df = pd.DataFrame({"item_id": test_index})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv(DIR + "submit.csv", index=False)
    # sub_df.to_csv(DIR + "train_act_pred00.csv", index=False)

    logger.info('Test End - Finish')
