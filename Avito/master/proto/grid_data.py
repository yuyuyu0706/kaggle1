import numpy as np
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from set_logger import logger

def rmse(label_y, pred_y):
    return np.sqrt(mean_squared_error(label_y, pred_y))

def gs(set_params, mode, dtrn_X, trn_y, dval_X, val_y, wl=None):
    if mode is None:
        print("mode is None")
        min_params = None
    elif mode == "lgb":
        min_params = lgb_gs(set_params, dtrn_X, trn_y, dval_X, val_y)
    elif mode == "xgb":
        min_params = xgb_gs(set_params, dtrn_X, trn_y, dval_X, val_y, wl)
    return min_params

def lgb_gs(set_params, dtrn_X, trn_y,  dval_X, val_X, val_y):
    min_score = 100
    for params in tqdm(list(ParameterGrid(set_params))):
        logger.debug('params:\n {}'.format(params))
        model = lgb.train(params, dtrn_X,
                          num_boost_round=1000,
                          valid_sets=[dval_X],
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

    return min_params

def xgb_gs(set_params, dtrn_X, trn_y,  dval_X, val_y, wl):
    min_score = 100
    for params in tqdm(list(ParameterGrid(set_params))):
        logger.debug('params:\n {}'.format(params))
        model = xgb.train(params, dtrn_X,
                          num_boost_round=1000, evals=wl,
                          early_stopping_rounds=100,
                          verbose_eval=50)

        pred = model.predict(dval_X,
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

    return min_params
