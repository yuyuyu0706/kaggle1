import numpy as np
import pandas as pd
from logging import getLogger

TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'
FEAT_DATA = '../input/aggregated_features.csv'
PERIOD_TRAIN_DATA = '../input/periods_train.csv'
PERIOD_TEST_DATA = '../input/periods_test.csv'
TRAIN_ACT_DATA_00 = '../input/train_act_split_0000.csv'
TRAIN_ACT_DATA_01 = '../input/train_act_split_0001.csv'
TRAIN_ACT_DATA_02 = '../input/train_act_split_0002.csv'

logger = getLogger(__name__)

def read_csv(path):
    logger.debug('enter')
    # df = pd.read_csv(path, index_col=["item_id"], parse_dates=["activation_date"])
    df1 = pd.read_csv(path, parse_dates=["activation_date"])
    df2 = read_feat_csv(FEAT_DATA)
    
    df1.drop_duplicates(subset='item_id', keep=False, inplace=True)
    df1 = pd.merge(df1, df2, on=['user_id'], how='left')
    df1['avg_days_up_user'] = df1['avg_days_up_user'].fillna(0).astype('uint32')
    df1['avg_times_up_user'] = df1['avg_times_up_user'].fillna(0).astype('uint32')
    df1['n_user_items'] = df1['n_user_items'].fillna(0).astype('uint32')
    
    df1.set_index('item_id', inplace=True)
    logger.debug('exit')
    del df2
    return df1

def read_csv_nrows(path, nrows):
    logger.debug('enter')
    # df = pd.read_csv(path, index_col=["item_id"], parse_dates=["activation_date"], nrows=nrows)
    df1 = pd.read_csv(path, parse_dates=["activation_date"], nrows=nrows)
    df2 = read_feat_csv_nrows(FEAT_DATA, nrows)
    
    df1.drop_duplicates(subset='item_id', keep=False, inplace=True)
    df1 = pd.merge(df1, df2, on=['user_id'], how='left')
    df1['avg_days_up_user'] = df1['avg_days_up_user'].fillna(0).astype('uint32')
    df1['avg_times_up_user'] = df1['avg_times_up_user'].fillna(0).astype('uint32')
    df1['n_user_items'] = df1['n_user_items'].fillna(0).astype('uint32')
    
    df1.set_index('item_id', inplace=True)
    logger.debug('exit')
    del df2
    return df1

def read_feat_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    logger.debug('exit')
    return df

def read_feat_csv_nrows(path, nrows):
    logger.debug('enter')
    df = pd.read_csv(path, nrows=nrows)
    logger.debug('exit')
    return df

def load_train_data(nrows=None):
    logger.debug('enter')
    if nrows is None:
        df = read_csv(TRAIN_DATA)
    else:
        df = read_csv_nrows(TRAIN_DATA, nrows)
    logger.debug('exit')
    return df

def load_test_data(nrows=None):
    logger.debug('enter')
    if nrows is None:
        df1 = read_csv(TEST_DATA)
    else:
        df1 = read_csv_nrows(TEST_DATA, nrows)
    logger.debug('exit')
    return df1

def load_period_train_data(nrows=None):
    logger.debug('enter')
    if nrows is None:
        df = read_csv(PERIOD_TRAIN_DATA)
    else:
        df = read_csv_nrows(PERIOD_TRAIN_DATA, nrows)
    logger.debug('exit')
    return df

def load_period_test_data(nrows=None):
    logger.debug('enter')
    if nrows is None:
        df = read_csv(PERIOD_TEST_DATA)
    else:
        df = read_csv_nrows(PERIOD_TEST_DATA, nrows)
    logger.debug('exit')
    return df

def load_train_act_data(nrows=None):
    logger.debug('enter')
    if nrows is None:
        df1 = read_csv(TRAIN_ACT_DATA_00)
        df2 = read_csv(TRAIN_ACT_DATA_01)
        df3 = read_csv(TRAIN_ACT_DATA_02)
        df = pd.concat([df1, df2, df3])
    else:
        df1 = read_csv_nrows(TRAIN_ACT_DATA_00, nrows)
        df2 = read_csv_nrows(TRAIN_ACT_DATA_01, nrows)
        df3 = read_csv_nrows(TRAIN_ACT_DATA_02, nrows)
        df = pd.concat([df1, df2, df3])
    logger.debug('exit')
    return df

if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())

