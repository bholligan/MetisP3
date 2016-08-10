from flask import Flask, jsonify, request, render_template

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle

#---------- MODEL IN MEMORY ----------------#

# Load the model from a pickle file
clf = pickle.load('#filename.pkl')
x_median = # Put array of features that are the median person
median_prob = clf.predict_proba(x_median)[1]

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

# Homepage
@app.route("/")
def index():
    """
    Homepage: serve our visualization page, index.html
    """
    return render_template("index.html")

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
    x = np.matrix(data["example"])
    person_prob = clf.predict_proba(x)[1]
    rel_score = person_prob/median_prob
    # Put the result in a nice dict so we can send it as json
    results = {"rel_score": rel_score}
    return jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
