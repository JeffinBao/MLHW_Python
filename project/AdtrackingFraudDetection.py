# coding: utf-8


import pandas as pd
import gc
import time
import pytz
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import sys


class AdtrackingFraudDetection:

    def __init__(self, train, test, model_name='xgb'):
        self.train = train
        self.test = test
        self.model_name = model_name

    def __split_data(self):
        self.y = self.train['is_attributed']
        self.train.drop(['is_attributed'], axis=1, inplace=True)
        gc.collect()

        self.submission = pd.DataFrame()
        self.submission['click_id'] = self.test['click_id'].astype('uint32')
        self.test.drop(['click_id'], axis=1, inplace=True)
        gc.collect()

    def __preprocess(self):
        self.__split_data()
        train_num_rows = len(self.train)

        # concat train and test dataset to do feature engineering
        combined_ds = pd.concat([self.train, self.test])

        # calculate frequency of ip
        start_time = time.time()
        temp = combined_ds['ip'].value_counts().reset_index(name='ip_count')
        temp.columns = ['ip', 'ip_count']
        combined_ds = combined_ds.merge(temp, on='ip', how='left')
        print('finish merge ip_count on ip:', time.time() - start_time)

        # calculate frequency of device
        start_time = time.time()
        temp = combined_ds['device'].value_counts().reset_index(name='device_count')
        temp.columns = ['device', 'device_count']
        combined_ds = combined_ds.merge(temp, on='device', how='left')
        print('finish merge device_count on device:', time.time() - start_time)

        # groupby ip-device on channel
        gb = combined_ds[['ip', 'device', 'channel']].groupby(by=['ip', 'device'])[
            ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_count'})
        combined_ds = combined_ds.merge(gb, on=['ip', 'device'], how='left')
        del gb
        gc.collect()

        # groupby ip-os on channel
        gb = combined_ds[['ip', 'os', 'channel']].groupby(by=['ip', 'os'])[['channel']].count().reset_index().rename(
            index=str, columns={'channel': 'ip_os_count'})
        combined_ds = combined_ds.merge(gb, on=['ip', 'os'], how='left')
        del gb
        gc.collect()

        # groupby device-os on channel
        gb = combined_ds[['device', 'os', 'channel']].groupby(by=['device', 'os'])[
            ['channel']].count().reset_index().rename(index=str, columns={'channel': 'device_os_count'})
        combined_ds = combined_ds.merge(gb, on=['device', 'os'], how='left')
        del gb
        gc.collect()

        # groupby ip-device-os on channel
        gb = combined_ds[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 'os'])[
            ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_count'})
        combined_ds = combined_ds.merge(gb, on=['ip', 'device', 'os'], how='left')
        del gb
        gc.collect()

        # drop ip, device, os since their counts have been calculated
        combined_ds.drop(['ip', 'device', 'os'], axis=1, inplace=True)
        gc.collect()

        # convert to CST timezone
        cst = pytz.timezone('Asia/Shanghai')
        start_time = time.time()
        combined_ds['click_time'] = pd.to_datetime(combined_ds['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(
            cst)

        # deal with time: extract hour
        combined_ds['hour'] = combined_ds['click_time'].dt.hour
        temp = combined_ds['hour'].value_counts().reset_index(name='hour_count')
        temp.columns = ['hour', 'hour_count']
        combined_ds = combined_ds.merge(temp, on='hour', how='left')
        print('finish merge hour_count on hour:', time.time() - start_time)

        # drop click_time and hour, since we have calculate the hour count
        combined_ds.drop(['click_time', 'hour'], axis=1, inplace=True)
        gc.collect()

        # split train and test data set after preprocessing
        self.train = combined_ds[:train_num_rows]
        self.test = combined_ds[train_num_rows:]

    def training(self):
        self.__preprocess()

        model = None
        if self.model_name == 'lr':
            model = self.__train_logistic_regression()
        elif self.model_name == 'dt':
            model = self.__train_decision_tree()
        elif self.model_name == 'xgb':
            model = self.__train_xgboost()
        elif self.model_name == 'rf':
            model = self.__train_random_forest()
        elif self.model_name == 'lgb':
            model = self.__train_lgb()

        return model

    def __train_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        logisticRegr = LogisticRegression(penalty='l2', C=1, tol=1e-5, solver='sag')
        logisticRegr.fit(self.train, self.y)

        return logisticRegr

    def __train_decision_tree(self):
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(class_weight=None, criterion='gini',
                                    max_features=None, max_leaf_nodes=None, min_samples_leaf=20,
                                    min_samples_split=50, min_weight_fraction_leaf=0.0,
                                    presort=False, random_state=100, splitter='best')
        dt.fit(self.train, self.y)
        return dt

    def __train_xgboost(self):
        scale_pos_weight = len(self.y[self.y == 0]) / len(self.y[self.y == 1])
        print('label=0:', len(self.y[self.y == 0]))
        print('label=1:', len(self.y[self.y == 1]))
        print('scale_pos_weight:', scale_pos_weight)

        # split train dataset into train and validation
        start_time = time.time()
        print('start training:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        X1, X2, y1, y2 = train_test_split(self.train, self.y, test_size=0.1)
        train_matrix = xgb.DMatrix(X1, y1)
        valid_matrix = xgb.DMatrix(X2, y2)
        watchlist = [(train_matrix, 'train'), (valid_matrix, 'valid')]
        params = {'eta': 0.25, 'tree_method': 'hist', 'grow_policy': 'lossguide', 'max_depth': 6, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'min_child_weight': 1, 'alpha': 3, 'objective': 'binary:logistic',
                  'scale_pos_weight': scale_pos_weight, 'eval_metric': 'auc'}

        model = xgb.train(params, train_matrix, 5, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
        print('finish training(time interval):', time.time() - start_time)
        return model

    def __train_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=12, max_depth=6, min_samples_leaf=100, max_features=0.5,
                                     bootstrap=False, n_jobs=-1, random_state=123)
        clf.fit(self.train, self.y)
        return clf

    def __train_lgb(self):
        X1, X2, y1, y2 = train_test_split(self.train, self.y, test_size=0.1)
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'num_leaves': 9,  # we should let it be smaller than 2^(max_depth)
            'max_depth': 5,  # -1 means no limit
            'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
            'max_bin': 100,  # Number of bucketed bin for feature values
            'subsample': 0.9,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'nthread': 8,
            'verbose': 0,
            'scale_pos_weight': 99.7,  # because training data is extremely unbalanced
        }

        dtrain = lgb.Dataset(X1.values, label=y1.values)
        dvalid = lgb.Dataset(X2.values, label=y2.values)

        MAX_ROUNDS = 1000
        EARLY_STOP = 50

        evals_results = {}

        model = lgb.train(lgb_params,
                          dtrain,
                          valid_sets=[dtrain, dvalid],
                          valid_names=['train', 'valid'],
                          evals_result=evals_results,
                          num_boost_round=MAX_ROUNDS,
                          early_stopping_rounds=EARLY_STOP,
                          verbose_eval=50,
                          feval=None)

        return model

    def __test_logistic_regression(self, model):
        predictions = model.predict_proba(self.test)
        self.submission['is_attributed'] = predictions[:, -1]

    def __test_decision_tree(self, model):
        predictions = model.predict_proba(self.test)
        self.submission['is_attributed'] = predictions[:, -1]

    def __test_xgboost(self, model):
        test_matrix = xgb.DMatrix(self.test)
        self.submission['is_attributed'] = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)

    def __test_random_forest(self, model):
        predictions = model.predict_proba(self.test)
        self.submission['is_attributed'] = predictions[:, 1]

    def __test_lgb(self, model):
        self.submission['is_attributed'] = model.predict(self.test)

    def predict(self, model):
        if self.model_name == 'lr':
            self.__test_logistic_regression(model)
        elif self.model_name == 'dt':
            self.__test_decision_tree(model)
        elif self.model_name == 'xgb':
            self.__test_xgboost(model)
        elif self.model_name == 'rf':
            self.__test_random_forest(model)
        elif self.model_name == 'lgb':
            self.__test_lgb(model)

        return self.submission


if __name__ == '__main__':
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    dtypes = {'ip': 'uint32',
              'app': 'uint16',
              'device': 'uint16',
              'os': 'uint16',
              'channel': 'uint16',
              'is_attributed': 'uint8',
              'click_id': 'uint32'}

    # read csv file, if read original train dataset, must subsample, use skiprows and nrows to select sub dataset
    start_time = time.time()
    print('start the program:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    train = pd.read_csv('../dataset/train_sample.csv', usecols=train_cols, dtype=dtypes)
    # train = pd.read_csv('../dataset/train.csv', usecols=train_cols, dtype=dtypes, skiprows=range(1, 104903891), nrows=80000000)
    # train = pd.read_csv('../dataset/train.csv', usecols=train_cols, dtype=dtypes)
    print('finish loading train data(time interval):', time.time() - start_time)

    start_time = time.time()
    test = pd.read_csv('../dataset/test.csv', usecols=test_cols, dtype=dtypes)
    print('finish loading test data(time interval):', time.time() - start_time)

    model_selection = sys.argv[1]
    # model_selection = 'lr'
    fraud_detection = AdtrackingFraudDetection(train, test, model_name=model_selection)
    model = fraud_detection.training()
    submission = fraud_detection.predict(model)
    submission.to_csv('submission.csv', index=False)
    print('exit the program:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
