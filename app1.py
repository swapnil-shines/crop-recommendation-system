from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

def recommendation(N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)

    # Get the probabilities for each crop class
    probabilities = model.predict_proba(transformed_features)[0]

    # Get the indices of the top N predictions
    top_n_indices = np.argsort(probabilities)[::-1][:top_n]

    # Map the indices to crop labels
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    top_crops = [crop_dict[index + 1] for index in top_n_indices]
    return top_crops

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    top_crops = recommendation(N, P, K, temp, humidity, ph, rainfall, top_n=3)

    result = f"Top crops to be cultivated: {', '.join(top_crops)}"
    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
