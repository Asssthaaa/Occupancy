import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import os

rooms_data=pd.read_csv("rooms_234.csv")
rooms_data.head(10)
rooms_data.rename(columns = {'Index':'Timestamp'}, inplace = True) 
rooms_data=rooms_data.drop(['Unnamed: 0','co2_log','light_log'],axis=1)
rooms_data['pir_classes'] = rooms_data['pir'].apply(lambda x: 1 if x > 0 else 0)
rooms_data

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

# Define the features to use in the model
selected_features = ['DayOfWeek', 'Month', 'Year', 'HourOfDay', 'Minute', 'Second']
rooms_scaler = StandardScaler()
# Function to preprocess data
def preprocess_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['HourOfDay'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute
    df['Second'] = df['Timestamp'].dt.second
    
    df[['temperature', 'humidity', 'co2', 'light']] = rooms_scaler.fit_transform(df[['temperature', 'humidity', 'co2', 'light']])
    return df

room_list=[413]
room_models = {}

for room_name in room_list:
    # Filter data for the current location
    rooms = rooms_data[rooms_data['room_ID'] == room_name].copy()

    rooms = preprocess_data(rooms)

    # Split the data into input features (X) and target variables (y)
    X = rooms[selected_features]
    y = rooms[['temperature', 'humidity', 'co2', 'light']]  # Multiple target variables

    split_index = int(len(rooms) * 0.8)  # 80-20 train-test split
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build the LSTM model
    model = Sequential([
        Bidirectional(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dense(4)  # Output layer for predicting multiple features
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mae')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))
        # Store the trained model
    room_models[room_name] = model

   

output_dir = "C://Users//Aastha Verma//room//all_models//"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for name, model in room_models.items():
    filename = os.path.join(output_dir, f'{name}.z')
    with open(filename, 'wb') as f:
        joblib.dump(model, f)

print("Models saved successfully.")

    