from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open('crop_recommendation/crop_model.pkl', 'rb'))
scaler = pickle.load(open('crop_recommendation/crop_scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        return render_template('index.html', prediction_text=f'Recommended Crop: {prediction[0]}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
