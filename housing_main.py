from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from models import Model, get_error
import xgboost as xgb
import numpy as np
import pandas as pd
import warnings
from sklearn import clone
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score


# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# This function is the main workhorse of this script. It does the following:
# 	1. Imports datasets and performs scaling (train and test csvs from data folder)
#	2. Calls all base learners being used and trains them.
#	3. Calls the stacker
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 1. Imports and scaling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Importing Roger's dataset
	X_train_large, y_train, X_predict_large = import_data(name_data = "roger")

	### Smaller dataset import (for linear)
	# X_train_linear, y_train, X_predict_linear = import_data(name_data = "peter")

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 2. Base learners are trained and predictions are made on the training set.
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	#Linear model parameters
	X_train = X_train_large
	X_predict = X_predict_large

	lasso = Model(model = "lasso")
	lasso_pred = lasso.train_validate(X_train, y_train)
	lasso_test_pred = lasso.train_predict(X_train, y_train, X_predict)

	elastic = Model(model = "elastic")
	elastic_pred = elastic.train_validate(X_train, y_train)
	elastic_test_pred = elastic.train_predict(X_train, y_train, X_predict)

	# krr = Model(model = "krr")
	# krr_pred = krr.train_validate(X_train, y_train)
	# krr_test_pred = krr.train_predict(X_train, y_train, X_predict)

	#Nonlinear model parameters
	X_train = X_train_large
	X_predict = X_predict_large

	rf = Model(model = "rf")
	rf_pred = rf.train_validate(X_train, y_train)
	rf_test_pred = rf.train_predict(X_train, y_train, X_predict)

	xgb = Model(model = "xgb")
	xgb_pred = xgb.train_validate(X_train, y_train)
	xgb_test_pred = xgb.train_predict(X_train, y_train, X_predict)

	# lgb = Model(model = "lgb")
	# lgb_pred = lgb.train_validate(X_train, y_train)
	# lgb_test_pred = lgb.train_predict(X_train, y_train, X_predict)

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 3. Stacking implementation for metamodel
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Putting all of the training set predictions together into one numpy array.
	X_stack_train = np.column_stack((lasso_pred, elastic_pred, rf_pred, xgb_pred)) #, krr_pred, xgb_pred, lgb_pred))
	X_stack_predict = np.column_stack((lasso_test_pred, elastic_test_pred, rf_test_pred, xgb_test_pred))#, krr__test_pred, xgb_test_pred, lgb_test_pred))

	# Initializing Stacker and giving it the inputs:
	# X_stack: combined base model predictions on training set
	# y_train: true housing prices of training set 
	# X_test: Kaggle test set to predict on
	stack = Stacker(model = "average")
	stack.stack_valid_predict(X_stack_train, y_train, X_stack_predict, output_csv = "no")


# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# This is a stacking class that can stack with either xgboost or average stacker.
# Later I will add best base and average stacking.
#
# ~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~
#	- __init__: this initializes the stacking object, either xgboost metamodel, or an averaging function
#				can be used
#	- stack: this performs the stacking process
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Stacker():
	def __init__(self, model = "xgb"):
		# Inputs:
		#	- model: this initializes the model

		self.model_name = model

		# Checking if the model parameter is a valid option.
		list_models = ["xgb", "average"]
		if model not in list_models:
			raise ValueError('Please give a model that exists in housing_main.py: class "mean_stacker!"')

		if model == "xgb":
			self.model = xgb.XGBRegressor(
					colsample_bytree=0.1,
					gamma=10.0,
					learning_rate=0.01,
					max_depth=5,
					min_child_weight=20,
					n_estimators=2000,                                                                  
					reg_alpha=0.5,
					reg_lambda=0.6,
					subsample=0.5,
					seed=42,
					silent=1)

	def stack_valid_predict(self, X_stack_train_full, y_stack_train_full, X_stack_predict, output_csv = "no"):
		# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
		# This function performs stacking on the training dataset
		#
		# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
		# Input:
		#	- X_train: This is the base models predictions, shape is [1450 x n_base_models]
		#	- y_train: This is the target dataset for the base model predictions, shape is [1450 x 1]
		#	- X_test: The Kaggle test dataset
		# Output:
		#	- y_pred: predictions on X_test
		#	- output_csv: (optional) This can be submitted to Kaggle competition
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Creating train/test split
		X_train_stack, X_test_stack, y_train_stack, y_test_stack = train_test_split(X_stack_train_full, y_stack_train_full, 
                                                       test_size=0.2, random_state=42)
		
		# Fitting the model chosen.
		if self.model_name != "average":

			# Turn off warnings
			warnings.filterwarnings('ignore')

			instance = clone(self.model)
			instance.fit(X_train_stack, y_train_stack)
			y_pred_stack = instance.predict(X_test_stack)

			# Removing the clone
			del instance

			rmsle = get_error(y_test_stack, y_pred_stack, type = "rmse")

			# Getting score for training set stacking...
			print("The rmsle on the stacking for {} stacking is {}".format(self.model_name, rmsle))

		else:
			y_pred_stack = np.mean(X_stack_train_full, axis = 1)

			rmsle = get_error(y_stack_train_full, y_pred_stack, type = "rmse")

			# Getting score for training set stacking...
			print("The rmsle on the stacking for {} stacking is {}".format(self.model_name, rmsle))

		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Now that we have an idea of how good the model is, we would like to train it on the full train set,
		# and predict on the Kaggle test set so we can generate a csv.

		# Checking if user input was valid.
		output_csv_list = ["yes", "no"]
		if output_csv not in output_csv_list:
			raise ValueError('Please give either a "yes" or "no" to the parameter "output_csv" in "stack_valid_predict"!')

		# Train/predict on Kaggle test dataset...
		if output_csv == "yes":
			if self.model_name != "average":
				instance = clone(self.model)
				instance.fit(X_stack_train_full, y_stack_train_full)
				y_pred = instance.predict(X_stack_predict)

				# Removing the clone
				del instance
			else:
				y_pred = np.mean(X_stack_predict, axis = 1)

			rmsle = '%.5f'%(rmsle)
			filename = "./Predictions/" + "CV_" + str(rmsle) + ".csv"
			df = pd.DataFrame(y_pred)
			df.to_csv(filename)

def optimize():
	lasso_BO = BayesianOptimization(lasso_func, {'alpha': (0.001, 1000)})
	lasso_BO.explore({'alpha': [0.001, 0.01, 0.1, 1, 10, 100]})
	lasso_BO.maximize(n_iter=30)
	print(lasso_BO.res['max'])


def lasso_func(alpha):
	X_train, y_train, X_predict = import_data(name_data = "roger")

	val = cross_val_score(Lasso(alpha = alpha, random_state = 2), X_train, y_train, cv=2).mean()
	return val


def import_data(name_data):

	# Below is roger's dataset 
	if name_data == "roger":
		train = pd.read_csv('./Data/train_120feats_Dense_OutlierFree_LogTransform.csv')
		test = pd.read_csv('./Data/test_119feats_Dense_OutlierFree_LogTransform.csv')

	# Below is peter's dataset
	if name_data == "peter":
		train = pd.read_csv('./Data/train_linear.csv')
		test = pd.read_csv('./Data/test_linear.csv')

	# Change dataframes to numpy arrays
	train = train.as_matrix()
	X_predict = test.as_matrix()

	# Assuming sales price is the last column...we separate out train_dummies
	# We don't need y_train here because it is the same for train_linear, so we'll get it there.
	last_col = train.shape[1]-1
	X_train = train[:,0:last_col]
	y_train = train[:,last_col]

	return X_train, y_train, X_predict



if __name__ == "__main__":
    # execute only if run as a script
    main()
    # optimize()



















