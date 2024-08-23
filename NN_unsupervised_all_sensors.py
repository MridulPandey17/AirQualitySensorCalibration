import os 
import sys
import pandas as pd 
import numpy as np
import pickle 
import json
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf

NUM_NEAREST_SENSORS = 10
NUM_LAST_DAYS = 12
FEAT_NUM = 50
EPOCHS = int(sys.argv[1])

val_scores = False
print(f"NUM_LAST_DAYS = {NUM_LAST_DAYS}, NUM_NEAREST_SENSORS = {NUM_NEAREST_SENSORS}, EPOCHS = {EPOCHS}")

model_save_path = f"models/NN/unsupervised_lcs_all_data_{NUM_LAST_DAYS}days_{EPOCHS}epochs_{NUM_NEAREST_SENSORS}sensors.h5"
scores_save_path = f"logs/NN_train_unsup_scores/avg_scores_NN_{NUM_LAST_DAYS}days_{EPOCHS}epochs.json"
lcs_data_path = "data/AQI_LCS_data_prep.csv"



model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=[FEAT_NUM], activation=None),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.8),  # Example dropout rate of 0.5
    
    tf.keras.layers.Dense(64, activation=None),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),  # Adjust dropout rate as needed

    tf.keras.layers.Dense(1)  
])
model.compile(
    optimizer='adam',
    loss='mean_squared_error',  
    metrics=['mean_absolute_error']
)
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     min_delta=0.001,
#     mode='min',
#     verbose=1,
#     restore_best_weights=True
# )


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

    ite = 0

    for train_sensor_index in tqdm(range(num_lcs_sensors)):

        data = lcs_data.drop(columns=sensor_column_group[train_sensor_index]) # dropping the true label data
        train_sensor_data = lcs_data[sensor_column_group[train_sensor_index]] # contains the true label for the data

        X = get_k_closest_sensors_data(train_sensor_index, lcs_data, NUM_NEAREST_SENSORS)

        for i in range(0, int(X.shape[1]/5), 5):
            X.iloc[:, i*5 + 1] = X.iloc[:, i*5 + 1] - train_sensor_data.iloc[:, 1]
            X.iloc[:, i*5 + 2] = X.iloc[:, i*5 + 2] - train_sensor_data.iloc[:, 2]
            X.iloc[:, i*5 + 3] = X.iloc[:, i*5 + 3] - train_sensor_data.iloc[:, 3]
            X.iloc[:, i*5 + 4] = X.iloc[:, i*5 + 4] - train_sensor_data.iloc[:, 4]

        

        if val_scores:

            X_train = X.iloc[-2*int(num_entries_to_use/3):, :] # using just the NUM_LAST_DAYS data for traininig the model    
            X_val = X.iloc[-num_entries_to_use:-2*int(num_entries_to_use/3), :] # using just the NUM_LAST_DAYS data for traininig the model    


            y_train = train_sensor_data.iloc[-2*int(num_entries_to_use/3):, 0].values 
            y_val = train_sensor_data.iloc[-num_entries_to_use:-2*int(num_entries_to_use/3), 0].values 
        
        else:

            X_train = X.iloc[-int(num_entries_to_use):, :]
            y_train = train_sensor_data.iloc[-int(num_entries_to_use):, 0].values 
        
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        if val_scores:
            scaler = StandardScaler()
            X_val = scaler.fit_transform(X_val)


        ite+=1

        model.fit(X_train, y_train,epochs=EPOCHS)

        if val_scores:
            results = {}

            y_pred = model.predict(X_train)

            results["Train_R2"] = r2_score(y_pred,y_train)
            results["Train_MAE"] = mean_absolute_error(y_pred,y_train)

            y_pred = model.predict(X_val)

            results["Val_R2"] = r2_score(y_pred,y_val)
            results["Val_MAE"] = mean_absolute_error(y_pred,y_val)

    if val_scores:

        results["Train_R2"]/=ite
        results["Train_MAE"]/=ite
        results["Val_R2"]/=ite
        results["Val_MAE"]/=ite

        with open(scores_save_path, "w") as f:
            json.dump(results, f, indent=4)
    
    model.save(model_save_path)