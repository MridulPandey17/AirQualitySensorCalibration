import os 
import sys
import pandas as pd 
import numpy as np
import pickle 
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# HYPERPARAMS : 

NUM_NEAREST_SENSORS = 10
NUM_LAST_DAYS = 12
n_estimators = 100

print(f"NUM_LAST_DAYS = {NUM_LAST_DAYS}, NUM_NEAREST_SENSORS = {NUM_NEAREST_SENSORS}")

regressor = RandomForestRegressor(n_estimators=n_estimators)

model_save_path = f"models/rf_reg/unsupervised_lcs_all_data_{NUM_LAST_DAYS}_nest_{n_estimators}.pkl"
lcs_data_path = "data/AQI_LCS_data_prep.csv"


def get_k_closest_sensors_data(sensor_index, df, k):
    '''
    df : has say n sensors (each sensor is taking 5 columns [pm, temp, rh, lat, long])
    of these n sensors we return a dataframe containing all the data for k nearest sensors excluding the given sensor
    
    '''
    
    column_groups = [df.columns[i*5:i*5+5] for i in range(int(df.shape[1]/5))]
    main_sensor_data = df[column_groups[sensor_index]] # the sensor for which we will return k nearest sensors
    data = df.drop(columns=column_groups[sensor_index]) # excluding the main_sensor
    
    ms_coord = (main_sensor_data.iloc[0, 3], main_sensor_data.iloc[0, 4]) # (lat, long) for main_sensor
    remain_sensors_coord = [(data.iloc[0, i*5+3], data.iloc[0, i*5+4]) for i in range(int(data.shape[1]/5))]
    
    remain_sensors_dist = [abs(ms_coord[0] - x) + abs(ms_coord[1] - y)  for (x, y) in remain_sensors_coord]
    # print(remain_sensors_dist)
    indices_list = list(range(len(remain_sensors_dist)))
    sorted_indices_list = sorted(indices_list, key = lambda i : remain_sensors_dist[i])
    top_k_indices_list = sorted_indices_list[:k] # these indices groups need to be included
    # print(top_k_indices_list)
    
    columns_to_return = []
    for index in top_k_indices_list:
        columns_to_return += column_groups[index].tolist()
    
    return df[columns_to_return]

if __name__ == "__main__":
    
    lcs_data = pd.read_csv(lcs_data_path)
    lcs_data = lcs_data.iloc[:, 2:]

    num_lcs_sensors = int((lcs_data.shape[1]) / 5)
    print(f"num_lcs_sensors = {num_lcs_sensors}")

    sensor_column_group = []
    for i in range(num_lcs_sensors):
        sensor_column_group.append(lcs_data.columns[i*5:i*5+5])
        
    num_entries_to_use = int(24 * NUM_LAST_DAYS)
        
    for train_sensor_index in tqdm(range(num_lcs_sensors)):
    
        data = lcs_data.drop(columns=sensor_column_group[train_sensor_index]) # dropping the true label data
        train_sensor_data = lcs_data[sensor_column_group[train_sensor_index]] # contains the true label for the data
        
        X = get_k_closest_sensors_data(train_sensor_index, lcs_data, NUM_NEAREST_SENSORS)
        
        for i in range(0, int(X.shape[1]/5), 5):
            X.iloc[:, i*5 + 1] = X.iloc[:, i*5 + 1] - train_sensor_data.iloc[:, 1]
            X.iloc[:, i*5 + 2] = X.iloc[:, i*5 + 2] - train_sensor_data.iloc[:, 2]
            X.iloc[:, i*5 + 3] = X.iloc[:, i*5 + 3] - train_sensor_data.iloc[:, 3]
            X.iloc[:, i*5 + 4] = X.iloc[:, i*5 + 4] - train_sensor_data.iloc[:, 4]
            
        X = X.iloc[-num_entries_to_use:, :] # using just the NUM_LAST_DAYS data for traininig the model    
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = train_sensor_data.iloc[-num_entries_to_use:, 0].values 
        
        regressor.fit(X, y)
        
        
    # saving the model     
    with open(model_save_path, "wb") as f: 
        pickle.dump(regressor, f)
        
    
        