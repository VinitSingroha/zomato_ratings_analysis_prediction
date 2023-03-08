import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import sklearn


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
locations = pd.read_csv("location.csv")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    # features[3] = int(locations.loc[locations['Str']==f'{features[3]}'].Int)
    features = [int(x) for x in features]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)