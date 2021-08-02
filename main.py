from flask import Flask, render_template, request
import json
import numpy as np
import pickle
from flask_sqlalchemy import SQLAlchemy
from gevent.pywsgi import WSGIServer

from DiabetesLinearReg import regressordiabetes
from HeartDiseasePred import regressorheart
from CarPricePred import regressorcar
from MedicalInsuranceCostPred import regressorinsurance
from Co2LinearReg import regressorcarbon
app = Flask(__name__)

local_server = True

with open("config.json", 'r') as c:
    params = json.load(c)['params']
# model = pickle.load(open('model1.pkl', 'rb'))

if local_server:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)


class Field(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    slug = db.Column(db.String(120), nullable=False)
    content = db.Column(db.String(120), nullable=False)


@app.route('/')
def home():
    field = Field.query.filter_by().all()
    return render_template("index.html",field=field)


@app.route('/diabetes', methods=["GET", "POST"])
def diabetes():
    return render_template('diabetes.html', params=params)


@app.route('/predict', methods=["GET", "POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressordiabetes.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('diabetes.html', params=params,prediction_text='CO2    Emission of the vehicle is :{}'.format(output))


@app.route('/predictdiabetes', methods=["GET", "POST"])
def predictdiabetes():
    str1=""
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressordiabetes.predict(final_features)
    output = round(prediction[0], 2)
    if output==1:
        str1="YES"
    else:
        str1="NO"

    return render_template('diabetes.html', params=params,prediction_text=' Do I have Diabetes? :{}'.format(str1))


@app.route('/carbon', methods=["GET", "POST"])
def carbon():
    return render_template('carbon.html', params=params)


@app.route('/predictcarbon', methods=["GET", "POST"])
def predictcarbon():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressorcarbon.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('carbon.html', params=params,prediction_text='CO2 Emission of the vehicle is :{}'.format(output))


@app.route('/heart', methods=["GET", "POST"])
def heart():
    return render_template('heart.html', params=params)


@app.route('/predictheart', methods=["GET", "POST"])
def predictheart():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressorheart.predict(final_features)
    output = round(prediction[0], 2)
    str1=""
    if output==1:
        str1="YES"
    else:
        str1="NO"

    return render_template('heart.html', params=params,prediction_text='Do I have Heart Disease? :{}'.format(str1))


@app.route('/car', methods=["GET", "POST"])
def car():
    return render_template('car.html', params=params)


@app.route('/predictcar', methods=["GET", "POST"])
def predictcar():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressorcar.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('car.html', params=params,prediction_text='Car Price is (in lakh) :{}'.format(output))


@app.route('/insurance', methods=["GET", "POST"])
def insurance():
    return render_template('insurance.html', params=params)


@app.route('/predictinsurance', methods=["GET", "POST"])
def predictinsurance():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressorinsurance.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('insurance.html', params=params,prediction_text='Medical Insurance Cost is :{}'.format(output))


@app.route('/about')
def about():
    return render_template('about.html', params=params)


if __name__ == '__main__':
    app.run(debug=True)
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()