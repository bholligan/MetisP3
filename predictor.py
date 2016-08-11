from flask import Flask, jsonify, request, render_template

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle

#---------- MODEL IN MEMORY ----------------#

# Load the model from a pickle file
with open('diabetes_xgboost.pkl', 'rb') as f:
    clf = pickle.load(f, encoding = 'Latin1')
with open('diabetes_med.pkl', 'rb') as f:
    meds = pickle.load(f, encoding = 'Latin1')
median_prob = clf.predict_proba(meds)[0][0]

with open('k_obesity_xgb.pkl', 'rb') as f:
    clf_k_obesity = pickle.load(f)
with open('k_obesity_meds.pkl', 'rb') as f:
    meds_k_obesity = pickle.load(f)
median_k_obesity = clf_k_obesity.predict_proba(meds_k_obesity)[0][1]

def change_df(df, s, a, c):
    df.iloc[0, 11] = s
    df.iloc[0, 14] = a
    df.iloc[0, 6] = c
    return df

def change_k_obesity(df, pov, P1_age, P1_prestige):
    df.iloc[0, 13] = pov
    df.iloc[0, 3] = P1_age
    df.iloc[0, 5] = P1_prestige
    return df

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)


# Homepage
@app.route("/")
def index():
    """
    Homepage: serve our visualization page, index.html
    """
    return render_template("index.html",
        median_prob = median_prob)

@app.route("/k_obesity/")
def kobesity():

    return render_template("k_obesity.html",
        median_prob = median_k_obesity)

# Get an example and return its score from the predictor model
@app.route("/score/", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = request.json
    sugar, age, cholesterol = data[0], data[1], data[2]
    df_u = change_df(meds, sugar, age, cholesterol)
    person_prob = clf.predict_proba(df_u)[0][0]
    print("Person P: {}".format(person_prob))
    print("Median P: {}".format(median_prob))
    rel_score = str(person_prob/median_prob)
    print("Rel Score: {}".format(rel_score))
    # Put the result in a nice dict so we can send it as json
    results = {"rel_score": rel_score}
    return jsonify(results)

@app.route("/score2/", methods=["POST"])
def score2():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = request.json
    pov, P1_age, P1_prestige = data[0], data[1], data[2]
    df_o = change_k_obesity(meds_k_obesity, pov, P1_age, P1_prestige)
    person_prob = clf_k_obesity.predict_proba(df_o)[0][1]
    print("Person P: {}".format(person_prob))
    print("Median P: {}".format(median_k_obesity))
    rel_score = str(person_prob/median_k_obesity)
    print("Rel Score: {}".format(rel_score))
    # Put the result in a nice dict so we can send it as json
    results = {"rel_score": rel_score}
    return jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
