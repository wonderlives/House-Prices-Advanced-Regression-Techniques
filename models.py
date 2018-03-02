from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import KFold 


# Summary:
# "model" builds a model from a variety of options.
# It can be trained and it can also predict.
class model():
	def __init__(self, alpha = .85, random_state = 1, model = "lasso"):
		if model == "lasso":
			self.model = Lasso(alpha = alpha, random_state = random_state))
		if model == "elastic":
			self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
		if model == "rf":
			self.model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
		if model == "krr":
			elf.model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
		if model == "xgb":
			self.model = xgb.XGBRegressor(
                colsample_bytree=0.1,
                gamma=10.0,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=20,
                n_estimators=7200,                                                                  
                reg_alpha=0.5,
                reg_lambda=0.6,
                subsample=0.5,
                seed=42,
                silent=1)
		if model == "lgb":
			self.model = LGBMRegressor(
				objective='regression',
				num_leaves=5,
                learning_rate=0.05,
                n_estimators=720,
                max_bin = 55,
                bagging_fraction = 0.8,
                bagging_freq = 5,
                feature_fraction = 0.2319,
                feature_fraction_seed=9,
                bagging_seed=9,
                min_data_in_leaf=6,
                min_sum_hessian_in_leaf=11)

	def fit_and_validate(self, X, y, n_folds =5):
		# Fits lasso to n_folds, computes error across all folds averaged and saves model.
		# Note that this fits on 80% of data in 5 folds
		# Inputs: 
		#	- X: this is the full training set, ~1450 observations
		#	- y: this the labels from the training set, ~1450 labels
		#	- n_folds: number of folds to run for validation

		# Either stratified or not
		skf = KFold(n_splits=n_folds)
		#skf = StratifiedKFold(n_splits=n_folds)

		log_error_sum = 0
		for train_index, test_index in skf.split(X, y):
    		X_train, X_test = X[train_index], X[test_index]
    		y_train, y_test = y[train_index], y[test_index]

			self.model.fit(X_train, y_train)
			y_pred = self.model.predict(X_test)

			# Anything less than $0 is put to 50,000
			mask = y_pred<0
    		y_pred[mask] = 50000

    		log_error_part = get_error(y_test, y_pred)
    		log_error_sum += log_error_part

    	print("Log error across all fold for lasso is {}".format(get_error(y_test, y_pred)))

	def predict(self, X)
		# Input:
		#	- any X to predict
		y_pred = self.model.predict(X)
		return y_pred


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