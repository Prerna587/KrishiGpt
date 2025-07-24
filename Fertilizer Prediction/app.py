from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load all saved models and encoders
model = pickle.load(open("Fertilizer Prediction/fertilizer_model.pkl", "rb"))
scaler = pickle.load(open("Fertilizer Prediction/fertilizer_scaler.pkl", "rb"))
soil_encoder = pickle.load(open("Fertilizer Prediction/soil_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("Fertilizer Prediction/crop_encoder.pkl", "rb"))
fertilizer_encoder = pickle.load(open("Fertilizer Prediction/fert_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil = request.form['soil']
        crop = request.form['crop']
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorous = float(request.form['phosphorous'])

        # Encode categorical inputs
        soil_encoded = soil_encoder.transform([soil])[0]
        crop_encoded = crop_encoder.transform([crop])[0]

        # Create input array
        input_data = np.array([[temperature, humidity, moisture, soil_encoded,
                                crop_encoded, nitrogen, potassium, phosphorous]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction_index = model.predict(input_scaled)[0]
        predicted_fertilizer = fertilizer_encoder.inverse_transform([prediction_index])[0]

        return render_template('index.html', prediction_text=f'Recommended Fertilizer: {predicted_fertilizer}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
