
#from lstm_model import room_models
#from lstm_model import rooms_scaler
import pickle
import numpy as np
room_id = '415'
with open(f"C://Users//Aastha Verma//room//room_models789//{room_id}.z",'rb') as f:
    modelh=pickle.load(f)
if modelh is None:
    print(f"Error: Model not found for room ID {room_id}")
else:
    print(modelh)


with open("C://Users//Aastha Verma//room//rooms_scaler.z",'rb') as f:
    rooms_scaler=joblib.load(f)

import pandas as pd 
from datetime import datetime 
def predict_features(timestamp_str, room_id):
    datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    #Convert timestamp string to datetime object
    datetime_obj = pd.to_datetime(datetime_obj)
    
    # Extract relevant features from the datetime object
    day_of_week = datetime_obj.dayofweek
    month = datetime_obj.month
    year = datetime_obj.year
    hour = datetime_obj.hour
    minute = datetime_obj.minute
    second = datetime_obj.second
    print(type(day_of_week))
    print(type(month))
    print(type(year))
    print(type(day_of_week))
    print(type(day_of_week))
    print(type(day_of_week))
    print(type(day_of_week))
    # Create input array for prediction
    input_array = input_array = np.array([[day_of_week, month, year, hour, minute, second]])  # Placeholder features
    print(input_array)
    input_array = input_array.reshape((input_array.shape[0], 1, input_array.shape[1]))  # Reshape input
    
    # Get the corresponding model for the given room ID
    
    
     # Get the corresponding model for the given room ID
    
    with open("C://Users//Aastha Verma//room//room_models789//{room_id}.z",'rb') as f:
        modelh=pickle.load(f)
    if modelh is None:
        print(f"Error: Model not found for room ID {room_id}")
        return None
    
    # Perform prediction
    predicted_features = modelh.predict(input_array)

    # Inverse transform the predicted features to get the actual values

    predicted_features = rooms_scaler.inverse_transform(predicted_features)
    
    # Extract individual feature values
    temperature_pred, humidity_pred, co2_pred, light_pred = predicted_features[0]

    # Check for NaN values in predicted features
    if np.isnan(temperature_pred) or np.isnan(humidity_pred) or np.isnan(co2_pred) or np.isnan(light_pred):
        print("Error: NaN values encountered in predicted features.")
        return None
    
    return {
        'temperature': temperature_pred,
        'humidity': humidity_pred,
        'co2': co2_pred,
        'light': light_pred
    }
