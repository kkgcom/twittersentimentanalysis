import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
vect = joblib.load('tweeter_vector.pkl')
model = joblib.load('tweeter_lr_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    inpt=request.form.get('tweet')
    vec= vect.transform([inpt])
    prediction = model.predict(vec)
    
    if round(prediction[0],2)==1.0:
        output = 'racist or sexist tweet'
    else:
        output = 'normal tweet'

    return render_template('index.html',prediction_text='The Tweet message is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)