# -*- coding: utf-8 -*-

import os
import json
import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
import features_reboot as features
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, make_scorer)

def raw_feature_iterator(file_paths):
	for path in file_paths:
		with open(path, "r") as fin:
			for line in fin:
				yield line

def vectorize(irow, raw_features_string, X_path, Y_path, extractor, nrows):
	raw_features = json.loads(raw_features_string)
	feature_vector = extractor.process_raw_features(raw_features)

	Y = np.memmap(Y_path, dtype=np.float32, mode="r+", shape=nrows)
	Y[irow] = raw_features["label"]

	X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
	#try:
	X[irow] = feature_vector
	#except:
	#	print(raw_features["sha256"])

def vectorize_unpack(args):
	return vectorize(*args)

def vectorize_subset(X_path, Y_path, raw_feature_paths, extractor, nrows):
	X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
	Y = np.memmap(Y_path, dtype=np.float32, mode="w+", shape=nrows)
	del X, Y

	pool = multiprocessing.Pool()
	argument_iterator = ((irow, raw_features_string, X_path, Y_path, extractor, nrows)
						for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
	for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
		pass

def create_vectorized_features(data_dir, feature_version=2):
	extractor = features.PEFeatureExtractor(feature_version)

	print("Vectorizing training set")
	X_path = os.path.join(data_dir, "X_train.dat")
	Y_path = os.path.join(data_dir, "Y_train.dat")

	raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(70)] # feature number
	nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
	vectorize_subset(X_path, Y_path, raw_feature_paths, extractor, nrows)

	print("Vectorizing test set")
	X_path = os.path.join(data_dir, "X_test.dat")
	Y_path = os.path.join(data_dir, "Y_test.dat")
	raw_feature_paths = [os.path.join(data_dir, "test_features_{}.jsonl".format(i)) for i in range(30)]
	nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
	vectorize_subset(X_path, Y_path, raw_feature_paths, extractor, nrows)

def read_vectorized_features(data_dir, subset=None, feature_version=2):
	if subset is not None and subset not in ["train", "test"]:
		return None

	extractor = features.PEFeatureExtractor(feature_version)
	ndim = extractor.dim
	X_train = None
	Y_train = None
	X_test = None
	Y_test = None

	if subset is None or subset == "train":
		X_train_path = os.path.join(data_dir, "X_train.dat")
		Y_train_path = os.path.join(data_dir, "Y_train.dat")
		Y_train = np.memmap(Y_train_path, dtype=np.float32, mode="r")
		N = Y_train.shape[0]
		X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))
		if subset == "train":
			return X_train, Y_train

	if subset is None or subset == "test":
		X_test_path = os.path.join(data_dir, "X_test.dat")
		Y_test_path = os.path.join(data_dir, "Y_test.dat")
		Y_test = np.memmap(Y_test_path, dtype=np.float32, mode="r")
		N = Y_test.shape[0]
		X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))
		if subset == "test":
			return X_test, Y_test

	return X_train, Y_train, X_test, Y_test

def read_metadata_record(raw_features_string):
	all_data = json.loads(raw_features_string)
	metadata_keys = {"sha256", "appeared", "label", "avclass"}
	return {k: all_data[k] for k in all_data.keys() & metadata_keys}

def create_metadata(data_dir):
	pool = multiprocessing.Pool()

	train_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(70)] # feature number
	train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))
	train_records = [dict(record, **{"subset": "train"}) for record in train_records]

	test_feature_paths = [os.path.join(data_dir, "test_features_{}.jsonl".format(i)) for i in range(30)]
	test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))
	test_records = [dict(record, **{"subset": "test"}) for record in test_records]

	all_metadata_keys = ["sha256", "appeared", "subset", "label", "avclass"]
	ordered_metadata_keys = [k for k in all_metadata_keys if k in train_records[0].keys()]
	metadf = pd.DataFrame(train_records + test_records)[ordered_metadata_keys]
	metadf.to_csv(os.path.join(data_dir, "metadata.csv"))
	return metadf

def read_metadata(data_dir):
	return pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)

def optimize_model(data_dir):
	X_train, Y_train = read_vectorized_features(data_dir, subset="train")

	train_rows = (Y_train != -1)

	X_train = X_train[train_rows]
	Y_train = Y_train[train_rows]

	score = make_scorer(roc_auc_score, max_fpr=5e-3)

	param_grid = {
		'boostring_type' : ['gbdt'],
		'objective': ['binary'],
		'num_iterations': [500, 1000],
		'learning_rate': [0.005, 0.05],
		'num_leaves': [512, 1024, 2048],
		'feature_fraction': [0.5, 0.8, 1.0],
		'bagging_fraction': [0.5, 0.8, 1.0]
	}

	model = lgb.LGBMClassifier(boosting_type="gbdt", n_jobs=-1, silent=True)

	progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train)

	grid = GridSearchCV(estimator=model, cv=progressive_cv, param_grid=param_grid, scoring=score, n_jobs=1, verbose=3)
	grid.fit(X_train, Y_train)

	return grid.best_params_

def train_model(data_dir, params={}, feature_version=2):
	params.update({"application": "binary"})

	X_train, Y_train = read_vectorized_features(data_dir, "train", feature_version)

	train_rows = (Y_train != -1)

	lgbm_dataset = lgb.Dataset(X_train[train_rows], Y_train[train_rows])
	lgbm_model = lgb.train(params, lgbm_dataset)

	return lgbm_model

def predict_sample(lgbm_model, file_data, sample_path, feature_version=2):
	extractor = features.PEFeatureExtractor(feature_version)
	feature = np.array(extractor.feature_vector(file_data, sample_path), dtype=np.float32)
	return lgbm_model.predict([feature])[0]