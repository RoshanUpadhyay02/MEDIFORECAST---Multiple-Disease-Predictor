import os
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import cv2

app = Flask(__name__)

def predict(values, dic):
    # diabetes
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        scaler = StandardScaler()
        values = scaler.fit_transform(values.reshape(-1, 1))
        return model.predict(values.reshape(1, -1))[0]

    # breast_cancer
    elif len(values) == 30:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = pd.DataFrame(values).T
        return model.predict(values)

    # heart disease
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # kidney disease
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # liver disease
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    
# Pages of Diseases

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

# Prediction of Diseases

@app.route("/diabetes_predict", methods=['GET', 'POST'])
def diabetesPredict():
    pred = None
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)
    
    return render_template('diabetes_predict.html', pred=pred)

@app.route("/breast_cancer_predict", methods=['GET', 'POST'])
def breastCancerPredict():
    pred = None
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)
    
    return render_template('breast_cancer_predict.html', pred=pred)

@app.route("/heart_predict", methods=['GET', 'POST'])
def heartPredict():
    pred = None
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)
    
    return render_template('heart_predict.html', pred=pred)

@app.route("/kidney_predict", methods=['GET', 'POST'])
def kidneyPredict():
    pred = None
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)
    
    return render_template('kidney_predict.html', pred=pred)

@app.route("/liver_predict", methods=['GET', 'POST'])
def liverPredict():
    pred = None
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)
    
    return render_template('liver_predict.html', pred=pred)

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    pred = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = tf.keras.models.load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    pred = None

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                uploaded_image = request.files['image']
                
                if allowed_file(uploaded_image.filename):
                    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    image_array = cv2.resize(image, (100, 100))
                    X = np.array([image_array])            
                    model = tf.keras.models.load_model("models/pneumonia.h5")
                    predictions = model.predict(X)
                    pred = np.argmax(predictions, axis=1)[0]
                else:
                    message = "Invalid file format. Please upload an image."
                    return render_template('pneumonia.html', message=message)
            else:
                message = "No image file uploaded. Please upload an image."
                return render_template('pneumonia.html', message=message)
        except Exception as e:
            message = "An error occurred while processing the image: " + str(e)
            return render_template('pneumonia.html', message=message)

    return render_template('pneumonia_predict.html', pred=pred)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp'}


# Prescription of Medicines

@app.route("/prescription_diabetes", methods=['GET', 'POST'])
def prescriptionDiabetes():
    return render_template('prescription_diabetes.html')

@app.route("/prescription_breast_cancer", methods=['GET', 'POST'])
def prescriptionBreastCancer():
    return render_template('prescription_breast_cancer.html')

@app.route("/prescription_heart", methods=['GET', 'POST'])
def prescriptionHeart():
    return render_template('prescription_heart.html')

@app.route("/prescription_kidney", methods=['GET', 'POST'])
def prescriptionKidney():
    return render_template('prescription_kidney.html')

@app.route("/prescription_liver", methods=['GET', 'POST'])
def prescriptionLiver():
    return render_template('prescription_liver.html')

@app.route("/prescription_malaria", methods=['GET', 'POST'])
def prescriptionMalaria():
    return render_template('prescription_malaria.html')

@app.route("/prescription_pneumonia", methods=['GET', 'POST'])
def prescriptionPneumonia():
    return render_template('prescription_pneumonia.html')

if __name__ == '__main__':
    app.run(debug = True)