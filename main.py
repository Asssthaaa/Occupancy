from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime
import pickle
from xgboost import XGBClassifier
from flask_cors import CORS
#from predict import predict_features
import pandas as pd 
from datetime import datetime 
import numpy as np 
import joblib
import pickle



with open("C://Users//Aastha Verma//room//rooms_scaler.z",'rb') as f:
    rooms_scaler=joblib.load(f)

    
# Load the XGBoost model from the pickle file                           
try:
    with open('C://Users//Aastha Verma//room//xgb_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found!")
    exit(1)

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'templates'
CORS(app)

# Expected features for prediction
expected_features = ["room_ID", "datetime"]

@app.route("/", methods=["GET"])
def my_website():
    return render_template("home.html")

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict_occupancy():
    if request.method == "POST":
        # Get data from the request
        data = request.get_json() 
        print("Received request data:", data)  # Add this line for debugging

        # Check if required fields are present and valid
        if not all(field in data for field in expected_features):
            return jsonify({"error": "Missing or invalid fields in request data"}), 400

        try:
            # Process data (convert datetime, etc.)
            room_id = int(data["room_ID"])
            datetime_str = data["datetime"]
            datetime_input = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
            datetime_input = datetime_input.strftime("%Y-%m-%d %H:%M:%S")
            print(datetime_input)


            #datetime_input = str(datetime_input)
            print(datetime_input)
            #datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            #print(datetime_obj)
            #Convert timestamp string to datetime object
            datetime_obj = pd.to_datetime(datetime_input)
            print(datetime_obj)
            #datetime_obj = timestamp_str
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
                
            with open(f"C://Users//Aastha Verma//room//room_models789//{room_id}.z",'rb') as f:
                modelh=pickle.load(f)
            if modelh is None:
                print(f"Error: Model not found for room ID {room_id}")
                return None
                
            # Perform prediction
            predicted_features = modelh.predict(input_array)

            print(predicted_features)# Inverse transform the predicted features to get the actual values

            predicted_features = rooms_scaler.inverse_transform(predicted_features)
                
            print(predicted_features)
            # Extract individual feature values
            temperature = predicted_features[0][0]
            print(temperature)
            humidity = predicted_features[0][1]
            print(humidity)
            co2 = predicted_features[0][2]
            light = predicted_features[0][3]

            # Check for NaN values in predicted features
            if np.isnan(temperature) or np.isnan(humidity) or np.isnan(co2) or np.isnan(light):
                print("Error: NaN values encountered in predicted features.")
                return None
                
            #other_features = {'temperature': temperature_pred, 'humidity': humidity_pred, 'co2': co2_pred, 'light': light_pred}
            datetime_unix = datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S").timestamp()
            print(datetime_unix)
            # Create DataFrame with processed data
            features_df = pd.DataFrame({
                "room_ID": [room_id],
                "Timestamp": [datetime_unix],
                "temperature": [temperature],
                "humidity": [humidity],
                "co2": [co2],
                "light": [light]   
            })

            # Make prediction
            print(features_df)
            prediction_proba = model.predict_proba(features_df.values)[0]
            print(prediction_proba)
            occupancy_status = "Occupied" if prediction_proba[1] > 0.5 else "Not occupied"
            probability = prediction_proba[1] * 100

            # Return JSON response
            print(prediction_proba)
            probability = float("{:.2f}".format(probability))

            return jsonify({"room_ID": room_id, "predicted_occupancy": occupancy_status, "probability": probability})

        except ValueError as e:
            return jsonify({"error": f"Error processing request: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()
