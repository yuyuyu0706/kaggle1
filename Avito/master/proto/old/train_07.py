import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle as pkl
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split
from load_data import load_train_data, load_test_data
from load_data import load_period_test_data
from grid_data import xgb_gs, lgb_gs
from set_logger import logger

ROW = None
DIR = './result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
XGBFMAP = DIR + 'xgb.fmap'

def rmse(label_y, pred_y):
    return np.sqrt(mean_squared_error(label_y, pred_y))

def create_feats_map(features):
    with open(XGBFMAP, 'w') as f:
        i = 0
        for feat in features:
            f.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

def run_lgb(trn_X, trn_y, val_X, val_y):
    lg_params = {
        "objective": "regression",
        "boosting": "gbdt",
        "metric": "rmse",
        "num_leaves": 128,      # [32, 48, 64, 128]
        "learning_rate": 0.07,  # [0.05, 0.07, 0.1, 0.2]
        "feature_fraction": 0.7,
        "bagging_freq": 5,
        "bagging_fraction": 0.7,
        "bagging_seed": 2018,
        "verbosity": -1
        }
    
    lg_trn = lgb.Dataset(trn_X, label=trn_y)
    lg_val = lgb.Dataset(val_X, label=val_y)
    
    logger.info('split.train: {}'.format(trn_X.shape))
    logger.info('split.valid: {}'.format(val_X.shape))

    # GridSearch
    # min_params = lgb_gs(lg_params, lg_trn, trn_y, lg_val, val_X, val_y)
    
    model = lgb.train(lg_params, lg_trn,
                      num_boost_round=5000,
                      valid_sets=[lg_val],
                      early_stopping_rounds=100,
                      verbose_eval=50)
    
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

    for i in range(len(df_tmp.index)):
        logger.debug('\t{0:20s} : {1:>10.6f}'.format(
                         df_tmp.ix[i, 0], df_tmp.ix[i, 1]))
    return model

def run_xgb(trn_X, trn_y, val_X, val_y):
    xg_params = {
        "max_depth": 8,           # [4, 6, 8, 12]
        "min_child_weight": 6,    # [4, 6, 8]
        "learning_rate": 0.1,     # [0.05, 0.075, 0.1, 0.2]
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0,
        }

    xg_trn = xgb.DMatrix(trn_X, label=trn_y)
    xg_val = xgb.DMatrix(val_X, label=val_y)
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
    create_feats_map(list(trn_X.columns[2:]))
    feat_i = model.get_fscore(fmap=XGBFMAP)
    
    df_tmp = pd.DataFrame(list(feat_i.items()), columns=['feat_n', 'feat_i'])
    df_tmp = df_tmp.sort_values(by=['feat_i'], ascending=False)
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['feat_i'] = df_tmp['feat_i'] / df_tmp['feat_i'].sum()
    
    for i in range(len(df_tmp.index)):
        logger.debug('\t{0:20s} : {1:>10.6f}'.format(
                            df_tmp.ix[i, 0], df_tmp.ix[i, 1]))
    return model

if __name__ == '__main__':
    logger.info('Start')

    train_df = load_train_data(nrows=ROW)
    logger.info('train load end {}'.format(train_df.shape))

    test_df = load_test_data(nrows=ROW)
    logger.info('test load end {}'.format(test_df.shape))

    # pr_train_df = load_period_train_data(nrows=ROW)
    # logger.info('period train load end {}'.format(pr_train_df.shape))

    # pr_test_df = load_period_test_data(nrows=ROW)
    # logger.info('period test load end {}'.format(pr_test_df.shape))

    # Feature Enginiering
    train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
    test_df["activation_weekday"] = test_df["activation_date"].dt.weekday
    # train_df["activation_month"] = train_df["activation_date"].dt.month
    # test_df["activation_month"] = test_df["activation_date"].dt.month
    train_df["description"] = train_df["description"].fillna(' ')
    test_df["description"] = test_df["description"].fillna(' ')
    train_df['description_len'] = train_df['description'].apply(lambda x: len(x.split()))
    test_df['description_len'] = test_df['description'].apply(lambda x: len(x.split()))
    train_df['title_len'] = train_df['title'].apply(lambda x: len(x.split()))
    test_df['title_len'] = test_df['title'].apply(lambda x: len(x.split()))

    logger.info('Data Preparation End')
    
    # Label encode the categorical variables
    cat_vars = ["region",
                "city",
                "user_id",
                # "title",
                # "description",
                "parent_category_name",
                "category_name",
                "user_type",
                "param_1",
                "param_2",
                "param_3"
                ]
    for col in tqdm(list(cat_vars)):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    # Train Data Split
    # train_y = train_df["deal_probability"].values
    # train_df, valid_df, train_y, valid_y = train_test_split(train_df,
    #                         train_y, test_size=0.20, shuffle=False, random_state=0)

    valid_df  = train_df[train_df['activation_date'] == '2017-03-15']
    valid_df1 = train_df[train_df['activation_date'] == '2017-03-16']
    valid_df  = valid_df.append(valid_df1, ignore_index=True)

    # Drop Rows
    train_df = train_df.drop(train_df.index[train_df['activation_date'] == '2017-03-15'])
    train_df = train_df.drop(train_df.index[train_df['activation_date'] == '2017-03-16'])

    # Drop collums
    cols_to_drop = ["item_id",
                    # "user_id",
                    "title",
                    "description",
                    "activation_date",
                    "image"
                    ]
    train_X = train_df.drop(cols_to_drop, axis=1)
    valid_X = valid_df.drop(cols_to_drop, axis=1)
    test_X = test_df.drop(cols_to_drop, axis=1)

    # Labels Copy
    train_y = train_df["deal_probability"].values
    valid_y = valid_df["deal_probability"].values
    test_id = test_df["item_id"].values

    # Labels Drop
    train_X = train_X.drop(["deal_probability"], axis=1)
    valid_X = valid_X.drop(["deal_probability"], axis=1)

    logger.info('Data Preparation End & Train start')

    # Trainning LightGBM
    # model = run_lgb(train_X, train_y, valid_X, valid_y)

    # Trainning XGBoost
    train_X = train_X.drop(['title_len'], axis=1)
    valid_X = valid_X.drop(['title_len'], axis=1)
    test_X = test_X.drop(['title_len'], axis=1)
    model = run_xgb(train_X, train_y, valid_X, valid_y)

    with open('./model_lgb.pkl', mode='wb') as f:
        pkl.dump(model, f)

    logger.info('Train End')
    logger.info('')
    logger.info('Test Start')

    # Test LightGBM
    # pred_test = model.predict(test_X, num_iteration=model.best_iteration)
 
    # Test XGBoost
    test_X = xgb.DMatrix(test_X)

    # with open('./model_xgb.pkl', mode='rb') as f:
    #    model = pkl.load(f)

    pred_test = model.predict(test_X, ntree_limit=model.best_ntree_limit)

    # Making Submmit Files
    pred_test[pred_test > 1] = 1
    pred_test[pred_test < 0] = 0

    sub_df = pd.DataFrame({"item_id": test_id})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv(DIR + "submit.csv", index=False)

    logger.info('Test End - Finish')
