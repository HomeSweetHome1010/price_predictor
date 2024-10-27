from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
import threading

encoder = None
rf = None
is_model_loaded = False

app = Flask(__name__)
CORS(app)

def load_model():
    global encoder, rf, is_model_loaded
    # Load the encoder and model
    try:
        with open('trans_encoder_new.pkl', 'rb') as file:
            encoder = pickle.load(file)

        with open('trans_predictor_new.pkl', 'rb') as file:
            rf = pickle.load(file)

        is_model_loaded = True
        print("Model and encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def home():
    global is_model_loaded
    # Check if the model is loaded
    if not is_model_loaded:
        # Start loading the model in a background thread
        threading.Thread(target=load_model).start()
        status_message = "Model is loading, please wait..."
    else:
        status_message = "Model is ready! You can now proceed to make predictions."

    # HTML content to render
    html_content = f"""
    <html>
        <head><title>Property Price Estimator API</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
            <h1>Property Price Estimator API</h1>
            <p>{status_message}</p>
            <p>Use the <strong>/predict</strong> endpoint to get a property price estimate.</p>
        </body>
    </html>
    """
    return render_template_string(html_content)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_price():
    if not is_model_loaded:
        return jsonify({'error': 'Model is still loading, please try again later.'}), 503
    
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
