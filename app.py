import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('specie.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    out_str = ""
    if output == 1:
        out_str = "Setosa"
    elif output == 2:
        out_str = "Verginica"
    elif output == 0:
        out_str = "Versicolor"

    return render_template('index.html',prediction_text="This is {}".format(out_str))


if __name__ == '__main__':
    app.run(debug=True,port=8000)

