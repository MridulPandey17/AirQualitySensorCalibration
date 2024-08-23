import os 
import sys
import pandas as pd 
import numpy as np
import pickle 
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
#from gpr_unsupervised_all_sensors import NUM_NEAREST_SENSORS, model_save_path, NUM_LAST_DAYS

# %%
import warnings
warnings.filterwarnings("ignore")

NUM_LAST_DAYS = 12
EPOCHS = 25
NUM_NEAREST_SENSORS = 12

#if(len(sys.argv)==4):
#    NUM_LAST_DAYS = sys.argv[1]
#    EPOCHS = sys.argv[2]
#    NUM_NEAREST_SENSORS = sys.argv[3]
#else:
#    raise KeyError('Expected 3 inputs')

#model_save_path = f'/Users/architaggarwal/Documents/Projects/AQI/Air-Quality-Index-Calibration/Air-Quality-Index-Calibration_Validation_Exp_Redo/models/NN/unsupervised_lcs_all_data_{NUM_LAST_DAYS}days_{EPOCHS}epochs_{NUM_NEAREST_SENSORS}sensors.h5'

model_save_path = '/Users/architaggarwal/Documents/Projects/AQI/Air-Quality-Index-Calibration/Air-Quality-Index-Calibration_Validation_Exp_Redo/models/NN/unsupervised_lcs_all_data_12days_25epochs_12sensors.h5'
# %%
def get_location_specific_lcs_data(location, df):
    
    '''
    location : string, the location for which you want the lcs sensors
    
    returns df containing the lcs for that location
    '''
    
    columns_to_return = []
    for col in df.columns: 
        if location in col:
            columns_to_return.append(col)
            
    return df[columns_to_return]

# %%
def get_location_specific_hcs_data(location_index, df):
    
    '''
    
    location_index : int, the index for which we want the hcs sensors data
    returns df containing that hcs
    
    '''
    
    columns_to_return = []
    for x in list(df.columns): 
        if int(x.split('_')[-1]) == int(location_index):
            columns_to_return.append(x)
            
    return df[columns_to_return]

# %%
def get_k_closest_sensors_data(input_loc, df, k):

    '''
    input_loc : pair of (lat, long) of hcs location
    df : has say n sensors (each sensor is taking 5 columns [pm, temp, rh, lat, long])
    of these n sensors we return a dataframe containing all the data for k nearest sensors 
    '''
    
    column_groups = [df.columns[i*5:i*5+5] for i in range(int(df.shape[1]/5))]
    sensors_coord = [(df.iloc[0, i*5+3], df.iloc[0, i*5+4]) for i in range(int(df.shape[1]/5))]
    
    sensors_dist = [abs(input_loc[0] - x) + abs(input_loc[1] - y)  for (x, y) in sensors_coord]
    
    indices_list = list(range(len(sensors_dist)))
    sorted_indices_list = sorted(indices_list, key = lambda i : sensors_dist[i])
    top_k_indices_list = sorted_indices_list[:k] # these indices groups need to be included
    # print(top_k_indices_list)
    
    columns_to_return = []
    for index in top_k_indices_list:
        columns_to_return += column_groups[index].tolist()
    
    return df[columns_to_return]

# %%
def normalize_prep_data(num_last_train_entries_to_use:int, use_first:bool , train_sensor_lcs_data, train_sensor_hcs_data):
    
    # train_sensor_lcs_data.shape = (510, 5*num_train_sensor_lcs_data)
    # 
    for i in range(0, int(train_sensor_lcs_data.shape[1]/5), 5):
        train_sensor_lcs_data.iloc[:, i*5+1] = train_sensor_lcs_data.iloc[:, i*5+1] - train_sensor_hcs_data.iloc[:, 4] # temp
        train_sensor_lcs_data.iloc[:, i*5+2] = train_sensor_lcs_data.iloc[:, i*5+2] - train_sensor_hcs_data.iloc[:, 3] # rh 
        train_sensor_lcs_data.iloc[:, i*5+3] = train_sensor_lcs_data.iloc[:, i*5+3] - train_sensor_hcs_data.iloc[:, 7] # lat
        train_sensor_lcs_data.iloc[:, i*5+4] = train_sensor_lcs_data.iloc[:, i*5+4] - train_sensor_hcs_data.iloc[:, 8] # long
        
    # for First : use from the starting    
    
    if use_first:
        train_X = train_sensor_lcs_data.values[:num_last_train_entries_to_use, :]
        train_y = train_sensor_hcs_data.values[:num_last_train_entries_to_use, 2] 
        
    else:
        # for Last : use from the last
        train_X = train_sensor_lcs_data.values[-num_last_train_entries_to_use:, :]
        train_y = train_sensor_hcs_data.values[-num_last_train_entries_to_use:, 2] 
    
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    
    return train_X, train_y

# %%
with open("data/hcsIndex2Name.json", "r") as f: 
    hcsIndex2Name = json.load(f)

with open("data/hcsName2Index.json", "r") as f: 
    hcsName2Index = json.load(f)

# %%
lcs_data = pd.read_csv('data/AQI_LCS_data_prep.csv')
lcs_data = lcs_data.iloc[:, 2:]


# %%
locations_to_test_loc = ['BHAGALPUR', 'KATIHAR', 'SAMASTIPUR', 'SIWAN', 'HAJIPUR']
location_to_test_loc_index_dict = {loc: i for i, loc in enumerate(locations_to_test_loc)}
locations_to_test_index = [hcsName2Index[x.lower()] for x in locations_to_test_loc]


# %%
hcs_data = pd.read_csv('data/hcsdatacleaned.csv')
hcs_data = hcs_data.iloc[:, 1:]

# %%
hcs_data_list = [get_location_specific_hcs_data(x, hcs_data) for x in locations_to_test_index]

# %%
def prepare_train_test(location_to_train: str, num_last_train_days: int, use_first: bool, num_train_sensor_lcs=NUM_NEAREST_SENSORS, num_hcs_data_list=hcs_data_list, lcs_data=lcs_data):
    
    '''
    location_to_train : string, location for which we want to train the model
    num_last_train_days : int, number of last days of data to train on
    
    returns X_train, X_test, y_train, y_test
    '''
    
    num_train_entries_to_use = int(24 * num_last_train_days)
    
    location_to_train_index = location_to_test_loc_index_dict[location_to_train]
    train_sensor_hcs_data = hcs_data_list[location_to_train_index] # we will be finetuning on this data
    test_sensor_hcs_data = hcs_data_list[:location_to_train_index] + hcs_data_list[location_to_train_index+1:] # this will be used for testing
    
    train_sensor_hcs_loc = (train_sensor_hcs_data.iloc[0, 7], train_sensor_hcs_data.iloc[0, 8])
    train_sensor_lcs_data = get_k_closest_sensors_data(train_sensor_hcs_loc, lcs_data, num_train_sensor_lcs)
    
    
    train_X, train_y = normalize_prep_data(num_train_entries_to_use, use_first, train_sensor_lcs_data, train_sensor_hcs_data)
    
    # now let's get our test set ready
    num_test_entries = int(lcs_data.shape[0]) # testing on all the data
    test_sensor_lcs_data = [get_k_closest_sensors_data((x.iloc[0, 7], x.iloc[0, 8]), lcs_data, num_train_sensor_lcs) for x in test_sensor_hcs_data]
    test_data = [normalize_prep_data(num_test_entries, use_first, test_sensor_lcs_data[i], test_sensor_hcs_data[i]) for i in range(len(test_sensor_hcs_data))]
    
    return train_X, train_y, test_data 
    # numpy_array, numpy_array, list of tuples (numpy_array, numpy_array) :
    #                                           (X_test, y_test) for each location in test_sensor_hcs_data

# %%
def give_locations_to_test(location: str, locations_to_test_loc=locations_to_test_loc):
    
    location_list = []
    for x in locations_to_test_loc: 
        if x != location: 
            location_list.append(x)    
    
    return location_list

# %%
def convert_to_float(final_results : dict) -> dict:
    for key in final_results.keys():
        results = final_results[key]
        for key in results.keys():
            results[key]['r2'] = results[key]['r2'].astype(np.float64)
            results[key]['mae'] = results[key]['mae'].astype(np.float64)
        final_results[key]=results
    return final_results

# %%

# %%

# %%
for num_train_last_days in range(1, 22):
    
    save_results_path = f"logs/leave_one_out_finetune_hcs/finetune_{num_train_last_days}_days/NN_{NUM_LAST_DAYS}.json"
    folder_path = "/".join(save_results_path.split('/')[:-1])
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    final_results = {}
    for location in locations_to_test_loc:
        
        print('-' * 100)
        print(f"finetuning_location = {location}")
        
        
        iter_test_locations = give_locations_to_test(location)
        X, y, test_data = prepare_train_test(location, num_last_train_days=num_train_last_days, use_first=True)
        # print(test_data[0][0].shape, test_data[0][1].shape)
        # sys.exit()
        
        #with open(model_save_path, "rb") as f:
        #    model = pickle.load(f)

        model = load_model(model_save_path)
            
        model.verbose = True
        
        
        # using validation set to check for over fitting
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
        # print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

    
        # sys.exit()
        X_train = tf.convert_to_tensor(np.asarray(X_train).astype('float32'))
        X_val = tf.convert_to_tensor(np.asarray(X_val).astype('float32'))
        y_train = tf.convert_to_tensor(np.asarray(y_train).astype('float32'))
        y_val = tf.convert_to_tensor(np.asarray(y_val).astype('float32'))

        # fitting the model to the training data
        model.fit(X_train, y_train)
        print('*' * 100)
        
        # getting validation scores
        y_val_hat = model.predict(X_val)
        #val_results = {"r2": r2_score(y_val, y_val_hat),"mae": mean_absolute_error(y_val, y_val_hat), "n_iter_finetune": model.n_iter_, "fit_status": model.fit_status_}
        val_results = {"r2": r2_score(y_val, y_val_hat),"mae": mean_absolute_error(y_val, y_val_hat)}

        
        y_test_hat_list = [model.predict(x) for x, _ in test_data]
        test_results = [(r2_score(y_test, y_test_hat), mean_absolute_error(y_test, y_test_hat)) for (_, y_test), y_test_hat in zip(test_data, y_test_hat_list) ] # (r2_score, mae) 
        
        iter_dict = {"val_results": val_results}
        for i in range(len(iter_test_locations)):
            iter_dict[iter_test_locations[i]] = {"r2": test_results[i][0], "mae": test_results[i][1]}
        
        final_results[location] = iter_dict   
        
    final_results = convert_to_float(final_results)
    with open(save_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
