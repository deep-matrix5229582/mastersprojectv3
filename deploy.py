from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

gender_map = {"F": 0, "M": 1}
bp_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
cholestol_map = {"HIGH": 0, "NORMAL": 1}
drug_map = {0: "DrugY", 3: "drugC", 4: "drugX", 1: "drugA", 2: "drugB"}

def predict_drug(Age, 
                 Sex, 
                 BP, 
                 Cholesterol, 
                 Na_to_K):
  
    model = joblib.load('drug_mod.pkl')
    Sex = gender_map[Sex]
    BP = bp_map[BP]
    Cholesterol = cholestol_map[Cholesterol]
    y_predict = model.predict([[Age, Sex, BP, Cholesterol, Na_to_K]])[0]
    return drug_map[y_predict] 

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Sex = request.form['GENDER']
        BP = request.form['BP']
        Cholesterol = request.form['Cholesterol']
        Na_to_K = float(request.form['natok'])
        drug = predict_drug(Age, Sex, BP, Cholesterol, Na_to_K)
    return render_template('predict.html', prediction = (jsonify(recommended_drug = drug)))

if __name__ == "__main__":
    app.run(host='0.0.0.0')