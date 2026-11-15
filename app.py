import joblib 
import numpy as np 
from preprocess import process_review
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('model.joblib')  #loading the saved model
vectorizer = joblib.load('vectorizer.joblib') #loading the saved TF-IDF Vectorizer

@app.route("/predict", methods=["POST"])  # Whenever user Posts data to "/predict", call this function
def predict():

    data = request.get_json()  # reading the incoming json data
    reviews = data['reviews']  # extract the revew from the dictionary
    process_rev = process_review(reviews) #preprocess the review posted
    process_rev = vectorizer.transform([process_rev]) # ectract features from the review
    prediction = model.predict(process_rev) #predict the sentiment

    return jsonify({"prediction":"positive" if prediction[0] == "1" else "negative"})  #Output the sentiment. jsonify is sued to convert python dict to json object

@app.route("/health", methods=['GET']) # used to check if the server is running. it'll output status:ok if server is running.
def health():
    return jsonify({"status":"ok"})



if __name__ == "__main__":

   import os 
   app.run(host = "0.0.0.0", port = int(os.environ.get("PORT", 5000)))
