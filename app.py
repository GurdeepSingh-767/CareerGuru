from flask import Flask, render_template, jsonify, request
import os
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)

# Load the model
filename = 'xgboost.sav'
model = pickle.load(open(filename, 'rb'))

# Define career dictionary
careers = {
    0: "Applications Developer",
    1: "Business Intelligence Analyst",
    2: "Business Systems Analyst",
    3: "CRM Business Analyst",
    4: "CRM Technical Developer",
    5: "Data Architect",
    6: "Database Administrator",
    7: "Database Developer",
    8: "Database Manager",
    9: "Design & UX",
    10: "E-Commerce Analyst",
    11: "Information Security Analyst",
    12: "Information Technology Auditor",
    13: "Information Technology Manager",
    14: "Mobile Applications Developer",
    15: "Network Engineer",
    16: "Network Security Administrator",
    17: "Network Security Engineer",
    18: "Portal Administrator",
    19: "Programmer Analyst",
    20: "Project Manager",
    21: "Quality Assurance Associate",
    22: "Software Developer",
    23: "Software Engineer",
    24: "Software Quality Assurance(QA) / Testing",
    25: "Software Systems Engineer",
    26: "Solutions Architect",
    27: "Systems Analyst",
    28: "Systems Security Administrator",
    29: "Technical Engineer",
    30: "Technical Services/Help Desk/Tech Support",
    31: "Technical Support",
    32: "UX Designer",
    33: "Web Developer",
}


# Assuming you know the number of features expected by your model
# num_of_features =  # Add the actual number of features expected by your model


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = []
    for x in request.form.values():
        if not x.isnumeric():
            response = {"status" : 500,"status_msg": "Some fields are empty !"}
            return jsonify(response)

        features.append(int(x))

 
    res = model.predict([features])
    predicted_role = careers[res[0]]
    response = {"status" : 200,"status_msg": "The best career for you is \n" + predicted_role}
    return jsonify(response)


# if __name__ == "__main__":
#     app.run()
