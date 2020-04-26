import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_train_data, load_test_data, load_period_train_data, load_period_test_data

logger = getLogger(__name__)

DIR = './result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

def run_lgb(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
        }

    lg_train = lgb.Dataset(train_X, label=train_y)
    lg_val = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lg_train, 1000, valid_sets=[lg_val],
                      early_stopping_rounds=100, verbose_eval=20, evals_result=evals_result)

    return model, evals_result

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

    # Feature Weekday
    train_df["activation_weekday"] = train_df["activation_date"].dt.weekday

    # Label encode the categorical variables #
    cat_vars = ["region", "city", "parent_category_name",
                "category_name", "user_type", "param_1", "param_2", "param_3"]
    for col in cat_vars:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    # Drop collums
    cols_to_drop = ["item_id", "user_id", "title",
                    "description", "activation_date", "image"]
    train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)

    # Splitting Data
    dev_X = train_X.iloc[:-200000,:]
    val_X = train_X.iloc[-200000:,:]

    dev_y = train_y[:-200000]
    val_y = train_y[-200000:]

    # Trainning Data
    logger.info('Train start')
    model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y)
    logger.info('Train end')

    # Test
    logger.info('Test Start')

    test_df["activation_weekday"] = test_df["activation_date"].dt.weekday
    test_id = test_df["item_id"].values

    test_X = test_df.drop(cols_to_drop, axis=1)
    pred_test = model.predict(test_X, num_iteration=model.best_iteration)

    # Making Submmit Files
    pred_test[pred_test>1] = 1
    pred_test[pred_test<0] = 0

    sub_df = pd.DataFrame({"item_id" : test_id})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv(DIR + "submit.csv", index=False)

    logger.info('Finish')

