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

def change_df(df, s, a, c):
    df.iloc[0, 12] = s
    df.iloc[0, 15] = a
    df.iloc[0, 8] = c
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
    df_u = meds.reindex_axis(sorted(meds.columns), axis = 1)
    df_u = change_df(df_u, sugar, age, cholesterol)
    #df = pd.read_json(data, orient = 'records')
    #df = df.reindex_axis(sorted(df.columns), axis =1)
    person_prob = clf.predict_proba(df_u)[0][0]
    print("Person P: {}".format(person_prob))
    print("Median P: {}".format(median_prob))
    rel_score = str(person_prob/median_prob)
    print("Rel Score: {}".format(rel_score))
    # Put the result in a nice dict so we can send it as json
    results = {"rel_score": rel_score}
    return jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
