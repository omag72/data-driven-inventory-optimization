from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join('models', 'model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    discount = request.form.get('discount')
    qty = request.form.get('qty')
    leadtime = request.form.get('leadtime')

    input_data = pd.DataFrame([[discount, qty, leadtime]], columns=['Discount Applied', 'Qty Sold', 'LeadTime'])

    prediction = model.predict(input_data)
    rounded_prediction = round(prediction[0])

    return render_template('index.html', prediction=rounded_prediction)

if __name__ == '__main__':
    app.run(debug=True)
