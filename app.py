from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/stock_price_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form
    try:
        data = {
            'MA10': float(request.form['MA10']),
            'MA50': float(request.form['MA50']),
            'Daily Return': float(request.form['DailyReturn']),
            'Volatility': float(request.form['Volatility']),
        }
    except ValueError:
        return jsonify({'error': 'Invalid input data'}), 400

    # Convert data to DataFrame and scale it
    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
