# 🩺 MEDIFORECAST - Multiple Disease Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red.svg">
  <img src="https://img.shields.io/badge/Machine%20Learning-Healthcare-success.svg">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg">
</p>

## 📌 Overview

**MEDIFORECAST** is a Machine Learning-powered healthcare application that predicts the likelihood of multiple diseases based on patient health parameters.

The project provides a user-friendly **Streamlit web interface**, allowing users to enter medical information and receive instant predictions using pre-trained machine learning models.

> **Disclaimer:** This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.

---

## ✨ Features

- 🩸 Diabetes Prediction
- ❤️ Heart Disease Prediction
- 🧠 Parkinson's Disease Prediction
- 💻 Interactive Streamlit Interface
- ⚡ Instant Prediction Results
- 📊 Easy-to-use Dashboard
- 🤖 Pre-trained Machine Learning Models

---

## 🛠 Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- NumPy
- Pandas

### Web Framework
- Streamlit

### Model Storage
- Pickle

---

## 📂 Repository Structure

```
MEDIFORECAST---Multiple-Disease-Predictor/

│
├── app.py
├── saved_models/
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
├── datasets/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🩺 Diseases Supported

| Disease | Prediction Type |
|----------|-----------------|
| Diabetes | Binary Classification |
| Heart Disease | Binary Classification |
| Parkinson's Disease | Binary Classification |

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/RoshanUpadhyay02/MEDIFORECAST---Multiple-Disease-Predictor.git

cd MEDIFORECAST---Multiple-Disease-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🔄 Workflow

1. Load trained ML models
2. User enters medical parameters
3. Input preprocessing
4. Model inference
5. Disease prediction
6. Display prediction result

---

## 📷 Screenshots

Add screenshots inside the `images/` folder.

Example:

```
images/

home.png

diabetes_prediction.png

heart_prediction.png

parkinsons_prediction.png
```

---

## 🎯 Future Improvements

- Kidney Disease Prediction
- Liver Disease Prediction
- Breast Cancer Prediction
- Lung Disease Prediction
- PDF Medical Report Generation
- User Authentication
- Patient History Database
- Explainable AI (SHAP/LIME)
- Cloud Deployment

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Roshan Upadhyay**

GitHub: https://github.com/RoshanUpadhyay02

---

⭐ If you found this project useful, please consider giving it a star!
