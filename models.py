from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import clone
import xgboost as xgb
import numpy as np
# import lightgbm as lgb
import tensorflow as tf
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import KFold 
import warnings


# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# This is class that you can create to instantiate a model, be it lasso or xgb.
#
# ~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~
#	- def train_validate(): for training and validating only on training set
# 	- def train_predict(): for training on test set and submitting on test set to Kaggle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Model():

	def __init__(self, alpha = .85, random_state = 1, model = "lasso"):
		self.model_name = model

		# Checking if the model parameter is a valid option.
		list_models = ["lasso", "elastic", "rf", "krr", "xgb", "lgb"]
		if model not in list_models:
			raise ValueError('Please give a model that exists in models.py!')

		# Creating the correct model.
		if model == "lasso":
			self.model = Lasso(alpha = .001, random_state = 42)
		if model == "elastic":
			self.model = ElasticNet(alpha=.005, l1_ratio=.9, random_state=42)
		if model == "rf":
			self.model = RandomForestRegressor(n_estimators = 40, max_depth = 10, random_state = 42)
		if model == "krr":
			self.model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
		if model == "xgb":
			self.model = xgb.XGBRegressor(
                colsample_bytree=0.1,
                gamma=10.0,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=10,
                n_estimators=2000,                                                                  
                reg_alpha=0.5,
                reg_lambda=0.6,
                subsample=0.5,
                seed=42,
                silent=1)
		# if model == "lgb":
		# 	self.model = lgb.LGBMRegressor(
		# 		objective='regression',
		# 		num_leaves=5,
  #               learning_rate=0.05,
  #               n_estimators=720,
  #               max_bin = 55,
  #               bagging_fraction = 0.8,
  #               bagging_freq = 5,
  #               feature_fraction = 0.2319,
  #               feature_fraction_seed=9,
  #               bagging_seed=9,
  #               min_data_in_leaf=6,
  #               min_sum_hessian_in_leaf=11)

	def train_validate(self, X, y, n_folds = 5):
		# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
		# Fits lasso to n_folds, computes error across all folds averaged and saves model.
		# Note that this fits on 80% of data in 5 folds
		#
		# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
		# Inputs: 
		#	- X: this is the full training set, ~1450 observations
		#	- y: this the labels from the training set, ~1450 labels
		#	- n_folds: number of folds to run for validation
		# Outputs:
		#	- y_pred_all: this is the predictions from each fold combined with each other
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Turn off warnings
		warnings.filterwarnings('ignore')

		# Either stratified or not
		skf = KFold(n_splits=n_folds)
		#skf = StratifiedKFold(n_splits=n_folds)

		y_pred_all = np.zeros(len(y))
		log_error_sum = 0

		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			# Cloning to avoid residual fits
			instance = clone(self.model)
			instance.fit(X_train, y_train)
			y_pred = instance.predict(X_test)

			# Removing the clone
			del instance

			# Anything less than $0 is put to 50,000
			mask = y_pred<0
			y_pred[mask] = 50000

			# Adding to error sum.
			log_error_part = get_error(y_test, y_pred)
			log_error_sum += log_error_part

			# Predicting on out of fold
			y_pred_all[test_index] = y_pred 

		print("Log error across all validation folds for {} is {}".format(self.model_name, get_error(y_test, y_pred, type = "rmse")))
		return y_pred_all

	def train_predict(self, X_train, y_train, X_predict):
		# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
		# This function will train on the entire training dataset and will then create predictions on the Kaggle test set.
		# This function is mean primarily for the purpose of submission.
		#
		# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
		# Input:
		#	- X_train: Full train dataset to train on
		#	- y_train: Full y train to train on
		#	- X_predict: Full test dataset (without labels)
		# Output:
		#	- y_pred: predictions on X
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		instance = clone(self.model)
		instance.fit(X_train, y_train)

		y_pred = instance.predict(X_predict)
		return y_pred

# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# get_error returns root mean square log error or root mean square error.
# rmsle is good for Kaggle comparison
# rmse is good for seeing the actual average error
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_error(y_true, y_pred, type = "rmsle"):
	assert len(y_true) == len(y_pred), "Length of prediction and test do not match!"

	list_types = ["rmsle","rmse"]
	assert type in list_types, "Error type you requested is not an option. Choose 'rmse' or 'rmsle' please."

	# Root mean square error (RMSE)
	if type == "rmse":
		return np.square(y_pred - y_true).mean() ** 0.5

	# Root mean square log error (RMSLE)
	if type == "rmsle":
		return np.square(np.log(y_pred) - np.log(y_true)).mean() ** 0.5

