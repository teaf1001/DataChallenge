# -*- coding: utf-8 -*-

import os
import json
import argparse
import init
from init import *

def main():
	prog = "train_model"
	description = "Train model for Data Challenge"
	parser = argparse.ArgumentParser(prog=prog, description=description)
	parser.add_argument("-v", "--featureversion", type=int, default=2, help="Model featrure version")
	parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
	parser.add_argument("--optimize", help="gridsearch to find best parameters", action="store_true")
	args = parser.parse_args()

	if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
		parser.error("{} is not found..(directory with raw feature files)".format(args.datadir))

	X_train_path = os.path.join(args.datadir, "X_train.dat")
	Y_train_path = os.path.join(args.datadir, "Y_train.dat")
	if not (os.path.exists(X_train_path) and os.path.exists(Y_train_path)):
		print("Create vetorized features")
		features.create_vectorized_features(args.datadir, args.featureversion)

	params = {
		"boosting" : "gbdt",
		"objective" : "binary",
		"num_iterations" : 2000,
		"learning_rate" : 0.03,
		"num_leaves" : 2048,
		"max_depth" : 15,
		"min_data_in_leaf" : 50,
		"feature_fraction" : 0.5
	}
	if args.optimize:
		params = init.optimize_model(args.datadir)
		print("Best parameters: ")
		print(json.dumps(params, indent=2))

	print("[+] Training Lightgbm model..")
	lgbm_model = init.train_model(args.datadir, params, args.featureversion)
	lgbm_model.save_model(os.path.join(args.datadir, "model.txt"))

if __name__ == "__name__":
	main()
