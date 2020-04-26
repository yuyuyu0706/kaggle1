import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = './result_tmp/'

if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'feature.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('Start')

    train_df = load_train_data(nrows=100)
    logger.info('train load end {}'.format(train_df.shape))

    print(train_df.head())

    test_df = load_test_data()
    logger.info('test load end {}'.format(test_df.shape))

    print(test_df.head())

    logger.info('Finish')

