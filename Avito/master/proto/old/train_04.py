import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = './result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

def rmse(label_y, pred_y):
    return np.sqrt(mean_squared_error(label_y, pred_y))

def run_lgb(train_X, train_y):
    lg_params = {
        "objective": ["regression"],
        "boosting": ["gbdt"],
        "metric": ["rmse"],
        "num_leaves": [128],      # [32, 48, 64, 128]
        "learning_rate": [0.07],  # [0.05, 0.07, 0.1, 0.2]
        "feature_fraction": [0.7],
        "bagging_freq": [5],
        "bagging_fraction": [0.7],
        "bagging_seed": [2018],
        "verbosity": [-1]
        }

    # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    # cv = KFold(n_splits=3, shuffle=True, random_state=0)

    trn_X, val_X, trn_y, val_y = train_test_split(train_X, train_y, test_size=0.20, shuffle=True, random_state=0)
    lg_trn = lgb.Dataset(trn_X, label=trn_y)
    lg_val = lgb.Dataset(val_X, label=val_y)
    
    logger.info('split.train: {}'.format(trn_X.shape))
    logger.info('split.valid: {}'.format(val_X.shape))
    
    min_score = 100
    
    for params in tqdm(list(ParameterGrid(lg_params))):
        logger.debug('params:\n {}'.format(params))
        model = lgb.train(params, lg_trn,
                            num_boost_round=1000,
                            valid_sets=[lg_val],
                            early_stopping_rounds=100,
                            verbose_eval=50)
        pred = model.predict(val_X,
                             num_iteration=model.best_iteration)
        sc_rmse = rmse(val_y, pred)

        if min_score > sc_rmse:
            min_score = sc_rmse
            min_params = params

        logger.debug('rmse: {}'.format(sc_rmse))
        logger.info('current min rmse: {}'.format(min_score))

    logger.info('')
    logger.info('Top min params:\n {}'.format(min_params))
    logger.info('Top min rmse: {}'.format(min_score))

    '''
    trn_y = trn_y.astype('int')
    val_y = val_y.astype('int')
    model = lgb.LGBMClassifier(**min_params)
    model.fit(trn_X, trn_y,
              eval_set=[(val_X, val_y)],
              early_stopping_rounds=10,
              eval_metric="rmse")
    '''

    model = lgb.train(min_params, lg_trn,
                      num_boost_round=5000,
                      valid_sets=[lg_val],
                      early_stopping_rounds=100,
                      verbose_eval=50)

    return model

def run_xgb(train_X, train_y):
    xg_params = {
        "max_depth": [8],         # [4, 6, 8]
        "min_child_weight": [6],  # [4, 6, 8]
        "learning_rate": [0.1],   # [0.05, 0.075, 0.1, 0.2]
        "colsample_bytree": [0.8],
        "colsample_bylevel": [0.8],
        "reg_alpha": [0],
        "seed": [0],
        }
    
    trn_X, val_X, trn_y, val_y = train_test_split(train_X, train_y, test_size=0.20, shuffle=True, random_state=0)
    xg_train = xgb.DMatrix(trn_X, label=trn_y)
    xg_val = xgb.DMatrix(val_X, label=val_y)
    watchlist = [(xg_train, 'train'), (xg_val, 'eval')]
    
    logger.info('split.train: {}'.format(trn_X.shape))
    logger.info('split.valid: {}'.format(val_X.shape))
    
    min_score = 100
    for params in tqdm(list(ParameterGrid(xg_params))):
        logger.debug('params:\n {}'.format(params))

        model = xgb.train(params, xg_train,
                            num_boost_round=1000,
                            evals=watchlist,
                            early_stopping_rounds=100,
                            verbose_eval=50)

        pred = model.predict(xg_val,
                             ntree_limit=model.best_ntree_limit)
        sc_rmse = rmse(val_y, pred)
        if min_score > sc_rmse:
            min_score = sc_rmse
            min_params = params

        logger.debug('rmse: {}'.format(sc_rmse))
        logger.info('current min rmse: {}'.format(min_score))

    logger.info('')
    logger.info('Top min params:\n {}'.format(min_params))
    logger.info('Top min rmse: {}'.format(min_score))

    model = xgb.train(min_params, xg_train,
                      num_boost_round=5000,
                      evals=watchlist,
                      early_stopping_rounds=100,
                      verbose_eval=50)

    return model

if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('Start')

    train_df = load_train_data()
    logger.info('train load end {}'.format(train_df.shape))

    test_df = load_test_data()
    logger.info('test load end {}'.format(test_df.shape))

    # Labels
    train_y = train_df["deal_probability"].values
    test_id = test_df["item_id"].values

    # Feature Weekday
    train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
    test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

    # Label encode the categorical variables
    cat_vars = ["region", "city", "parent_category_name",
                "category_name", "user_type", "param_1", "param_2", "param_3"]
    for col in tqdm(list(cat_vars)):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    
    # Drop collums
    cols_to_drop = ["item_id", "user_id", "title",
                    "description", "activation_date", "image"]
    train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
    test_X = test_df.drop(cols_to_drop, axis=1)

    # Trainning LightGBM
    # logger.info('Train start')
    # model = run_lgb(train_X, train_y)
    # model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y)

    # Trainning XGBoost
    logger.info('Train start')
    model = run_xgb(train_X, train_y)

    # model, evals_result = run_xgb(dev_X, dev_y, val_X, val_y)
    logger.info('Train End')
    logger.info('')

    # Test LightGBM
    # logger.info('Test Start')
    # pred_test = model.predict(test_X, num_iteration=model.best_iteration)
 
    # Test XGBoost
    logger.info('Test Start')
    test_X = xgb.DMatrix(test_X)
    pred_test = model.predict(test_X, ntree_limit=model.best_ntree_limit)

    # Making Submmit Files
    pred_test[pred_test > 1] = 1
    pred_test[pred_test < 0] = 0

    sub_df = pd.DataFrame({"item_id": test_id})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv(DIR + "submit.csv", index=False)

    logger.info('Test End - Finish')

