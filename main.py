from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.base import clone
import os
from sklearn.model_selection import train_test_split
import re
import numpy as np
from flask import Response
import json

# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route('/train', methods=['POST'])
def retrain():
    # If a form is submitted
    if request.method == "POST":
        # Unpickle classifier
        clf = joblib.load("rfc3.pkl")
        clf_retrained = clone(clf)
        
        # Get values through input bars
        variables = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges', 'TotalCharges',"Churn"]
        
        # Put inputs to dataframe
        datos = request.get_json()
        df = pd.DataFrame(datos, columns = variables)
        df2 = pd.read_json("DataSet_Entrenamiento_v1.json")
        df.TotalCharges = pd.to_numeric(df.TotalCharges)
        y = df.Churn
        X = df.drop("Churn",axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        files = os.listdir("modelos_reentrenados")
        clf_retrained.fit(X_train,y_train)

        files = list(map(lambda x: 0 if x.startswith('.') else int(re.findall(r'\d+', x)[0]), files))
        actual = max(files)+1
        joblib.dump(clf_retrained, 'modelos_reentrenados/v{}.pkl'.format(actual))
        score_retrained = clf_retrained.score(X_test, y_test)
        score = clf.score(X_test, y_test)

        return {"ROC AUC modelo original":score,"ROC AUC modelo reentrenado" : score_retrained}
    else:
        return {"STATUS":400}

        
@app.route('/predict', methods=[ 'POST'])
def make_prediction():
    
     # If a form is submitted
    if request.method == "POST":
        try:
            version = request.args.to_dict()
            if len(version) == 0:
                files = os.listdir("modelos_reentrenados")
                files = list(map(lambda x: 0 if x.startswith('.') else int(re.findall(r'\d+', x)[0]), files))
                actual = max(files)
                if actual == 0:
                    clf = joblib.load("rfc3.pkl")
                else:
                    clf = joblib.load("modelos_reentrenados/v{}.pkl".format(actual))
            else:
                clf = joblib.load("modelos_reentrenados/v{}.pkl".format(version['version']))
        # Unpickle classifier
            
            # Get values through input bars
            variables = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                        'MonthlyCharges', 'TotalCharges']
            
            # Put inputs to dataframe
            datos = request.get_json()
            df = pd.DataFrame(datos, columns = variables)
            df.TotalCharges = pd.to_numeric(df.TotalCharges)
            z = clf.predict_proba(df)
            labels = np.argmax(z, axis=1)
            classes = clf.classes_
            labels = [classes[i] for i in labels]
            return list(zip(np.max(z,axis=1),labels))
        except Exception as e:
            return e
    else:
        prediction = ""
        
    return {"STATUS":200}

@app.route('/models', methods=['GET'])
def get_models():
    files = os.listdir("modelos_reentrenados")
    files.remove(".DS_Store")
    return files

# Running the app
if __name__ == '__main__':
    app.run(debug = True)