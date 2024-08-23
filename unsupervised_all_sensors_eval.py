import os 
import sys
import pandas as pd 
import numpy as np
import pickle 
import json
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from gpr_unsupervised_all_sensors import NUM_NEAREST_SENSORS

# HYPERPARAMS :

NUM_LAST_DAYS = 12
load_test_data_path = "data/hcs_test_data_dict.pickle"

## MODEL PATH

# GPR

# model_path = f"models/gpr_model_5/unsupervised_lcs_all_data_{NUM_LAST_DAYS}.pkl"
# model_name = model_path.split("/")[1].split("_")[0]
# results_save_path = f"logs/no_finetuning_unsupervised_lcs/{model_name}_scores_{NUM_LAST_DAYS}.json"

# Random Forest Regressor

# n_estimators = 100
# model_path = f"models/rf_reg/unsupervised_lcs_all_data_{NUM_LAST_DAYS}_nest_{n_estimators}.pkl"
# model_name = model_path.split("/")[1]
# results_save_path = f"logs/no_finetuning_unsupervised_lcs/{model_name}_nest_{n_estimators}_scores_{NUM_LAST_DAYS}.json"

# Lasso Regression

alpha = 10
model_path = f"models/lasso_reg/unsupervised_lcs_all_data_{NUM_LAST_DAYS}_alpha_{alpha}.pkl"
model_name = model_path.split("/")[1]
results_save_path = f"logs/no_finetuning_unsupervised_lcs/{model_name}_alpha_{alpha}_scores_{NUM_LAST_DAYS}.json"

# SVR
# kernel = "poly"
# model_path = f"models/svr/unsupervised_lcs_all_data_{NUM_LAST_DAYS}_kernel_{kernel}.pkl"
# model_name = model_path.split("/")[1]
# results_save_path = f"logs/no_finetuning_unsupervised_lcs/{model_name}_kernel_{kernel}_scores_{NUM_LAST_DAYS}.json"

# Loading the model

with open(model_path, "rb") as f: 
    model = pickle.load(f)
    
# loading the test data

with open(load_test_data_path, "rb") as f: 
    test_data_dict = pickle.load(f)
    
if __name__ == "__main__":
    
    no_finetuning_results = {}
    for loc, (X_test, y_test) in test_data_dict.items():
        
        y_test_hat = model.predict(X_test)
        no_finetuning_results[loc] = {'r2_score': r2_score(y_test, y_test_hat), "mae_score": mean_absolute_error(y_test, y_test_hat)}
        
        print(y_test_hat.shape, y_test.shape)

    with open(results_save_path, 'w') as f:
        json.dump(no_finetuning_results, f, indent=4)

