from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce

app = Flask(__name__)
CORS(app)

# Load the encoder and model
with open('trans_encoder_new.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('trans_predictor_new.pkl', 'rb') as file:
    rf = pickle.load(file)


@app.route('/')
def home():
    return "Property Price Estimator API"


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Extract data from request JSON
        data = request.json

        # Map room value
        room_value_mapping = {
            'Studio': 1,
            '1 B/R': 2,
            '2 B/R': 3,
            '3 B/R': 5,
            'Office': 4,
            'Others': 6
        }

        room_value = room_value_mapping.get(data.get('room_value', 'Studio'), 1)
        has_parking = 1 if data.get('has_parking') == 'Yes' else 0

        # Prepare the input query
        input_data = pd.DataFrame({
            'property_usage_en': [data['property_usage_en']],
            'property_type_en': [data['property_type_en']],
            'reg_type_en': [data['reg_type_en']],
            'area_name_en': [data['area_name_en']],
            'nearest_metro_en': [data['nearest_metro_en']],
            'room_value': [room_value],
            'has_parking': [has_parking],
            'procedure_area': [data['procedure_area']],
            'trans_group_en': [data['trans_group']]
        })

        # Apply the encoder to the input query
        query_encoded = encoder.transform(input_data)

        # Align features with the model's expected input
        query_encoded = query_encoded.reindex(columns=rf.feature_names_in_, fill_value=0)

        # Make prediction
        prediction = np.exp(rf.predict(query_encoded))

        # Return the predicted price
        return jsonify({'predicted_price': f"AED {int(prediction[0]):,}"})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
