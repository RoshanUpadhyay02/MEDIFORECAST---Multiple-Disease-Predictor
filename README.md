# 🩺 MEDIFORECAST — Multiple Disease Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-black.svg">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg">
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-red.svg">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
</p>

## 📌 Overview

**MEDIFORECAST** is a machine-learning and deep-learning based healthcare web application built with **Flask**. It provides prediction interfaces for multiple diseases using both structured patient information and medical images.

The project combines traditional machine-learning classifiers for tabular healthcare data with deep-learning image classification models for malaria blood-smear images and chest X-ray images.

The application provides separate pages for each supported disease, collects the required input through HTML forms or image uploads, loads pre-trained models, and displays the resulting prediction.

> ⚠️ **Disclaimer:** MEDIFORECAST is an educational and research project. Its predictions should not be considered a medical diagnosis or a substitute for evaluation by a qualified healthcare professional.

---

## ✨ Features

* 🩸 Diabetes prediction from patient health parameters
* 🎗️ Breast cancer prediction from diagnostic measurements
* ❤️ Heart disease prediction from cardiovascular parameters
* 🫘 Kidney disease prediction from clinical parameters
* 🧪 Liver disease prediction from liver-function measurements
* 🦠 Malaria classification from blood-smear images
* 🫁 Pneumonia classification from chest X-ray images
* 🌐 Flask-based web interface
* 🤖 Pre-trained machine-learning models
* 🧠 TensorFlow/Keras image-classification models
* 📊 Exploratory data analysis and model-comparison notebooks
* 📁 Separate datasets, notebooks, models, templates, and test images

---

# 🩺 Supported Predictions

| Disease        | Input                                 | Model Type                      |
| -------------- | ------------------------------------- | ------------------------------- |
| Diabetes       | 8 clinical parameters                 | Gradient Boosting Classifier    |
| Breast Cancer  | 30 diagnostic features                | Decision Tree Classifier        |
| Heart Disease  | 13 clinical/cardiovascular parameters | Random Forest Classifier        |
| Kidney Disease | 24 clinical parameters                | Random Forest Classifier        |
| Liver Disease  | 10 clinical/liver-function parameters | SVM                             |
| Malaria        | Blood-smear image                     | VGG19-based deep-learning model |
| Pneumonia      | Chest X-ray image                     | InceptionResNetV2-based model   |

---

# 🤖 Machine Learning Models

## 1. Diabetes Prediction

The diabetes model is trained using the **Pima Indians Diabetes Dataset**, containing **768 records** and 8 input features.

### Input Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

The notebook performs:

* Missing-value handling
* Replacement of invalid zero values
* Outcome-based median imputation
* Outlier handling for insulin
* Train/test splitting
* StandardScaler normalization
* Comparison of multiple classification algorithms

Algorithms evaluated include:

* Logistic Regression
* K-Nearest Neighbors
* Support Vector Machine
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The saved model is a **Gradient Boosting Classifier**.

The notebook reports a test accuracy of approximately **92.64%** for this model on its recorded train/test split.

---

## 2. Breast Cancer Prediction

The breast-cancer dataset contains **569 samples** and 30 diagnostic features.

The original diagnosis labels are converted to binary values:

* `0` — Benign
* `1` — Malignant

The model uses measurements including:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

These measurements are available for:

* Mean
* Standard error
* Worst-case measurements

Multiple models were evaluated:

* Logistic Regression
* Support Vector Machine
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The repository saves a **Decision Tree Classifier** as `models/breast_cancer.pkl`.

### Important evaluation note

The breast-cancer notebook evaluates the models on the full transformed dataset rather than using the held-out test set for the reported comparison. Therefore, the recorded **100% accuracy should not be interpreted as an independent test-set accuracy**.

---

## 3. Heart Disease Prediction

The heart-disease dataset contains **1,025 records** with 13 input features and a binary target.

### Input Features

* Age
* Sex
* Chest Pain Type (`cp`)
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression (`oldpeak`)
* Slope
* Number of Major Vessels (`ca`)
* Thalassemia (`thal`)

The notebook evaluates:

* Logistic Regression
* K-Nearest Neighbors
* SVM
* Decision Tree
* Tuned Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The saved model is a **Random Forest Classifier** with the following configuration:

* `n_estimators = 180`
* `max_depth = 7`
* `max_features = sqrt`
* `min_samples_leaf = 2`
* `min_samples_split = 4`

The notebook reports approximately **98.38% test accuracy** for this Random Forest model on its recorded 70/30 train/test split.

---

## 4. Kidney Disease Prediction

The kidney dataset contains **400 records** with clinical and laboratory measurements.

The preprocessing includes:

* Cleaning inconsistent categorical values
* Handling missing values
* Categorical feature encoding
* Train/test splitting
* StandardScaler normalization

The notebook compares:

* Logistic Regression
* SVM
* Decision Tree
* Random Forest
* XGBoost
* Gradient Boosting

The saved model is a **Random Forest Classifier**.

The notebook reports approximately **97.5% test accuracy** for the saved Random Forest model on its recorded train/test split.

---

## 5. Liver Disease Prediction

The liver dataset contains **583 records** and 10 input features.

### Input Features

* Age
* Gender
* Total Bilirubin
* Direct Bilirubin
* Alkaline Phosphatase
* Alamine Aminotransferase
* Aspartate Aminotransferase
* Total Proteins
* Albumin
* Albumin/Globulin Ratio

The notebook performs:

* Duplicate removal
* Gender label encoding
* Train/test splitting
* StandardScaler normalization
* Model comparison

Models evaluated include:

* Logistic Regression
* K-Nearest Neighbors
* SVM
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The saved model is an **SVM classifier**.

The notebook records approximately **71.18% test accuracy** for the saved SVM configuration.

---

# 🧠 Deep Learning Models

## 6. Malaria Image Classification

The malaria component uses blood-smear images to classify samples into two categories:

* Healthy
* Infected

The notebook experiments with **VGG19** and transfer-learning/fine-tuning techniques.

The final model uses a VGG19-based architecture with additional fully connected layers and fine-tuning of deeper convolutional layers.

The final recorded evaluation in the notebook reports:

* Validation accuracy: approximately **95.72%**
* Test accuracy: approximately **96.04%**

The test set contains **2,756 images**, with equal representation of healthy and infected classes in the reported evaluation.

The trained model is stored as:

```text
models/malaria.h5
```

---

## 7. Pneumonia Image Classification

The pneumonia component uses chest X-ray images.

The notebook uses **InceptionResNetV2** with additional dense layers.

The architecture includes:

```text
InceptionResNetV2
        ↓
Flatten
        ↓
Dense(1024, ReLU)
        ↓
Dense(512, ReLU)
        ↓
Output Layer
```

The notebook's initial model uses a 5-class output during experimentation, while the later pneumonia evaluation works with the two classes:

* NORMAL
* PNEUMONIA

The recorded evaluation on the 624-image test set reports approximately:

**99.04% accuracy**

with the classification report showing approximately:

* Normal recall: 97%
* Pneumonia recall: 100%

The trained model is stored as:

```text
models/pneumonia.h5
```

---

# 🌐 Web Application

The application is implemented using **Flask**, not Streamlit.

The main application is:

```text
app.py
```

The Flask application provides routes for:

```text
/
 /diabetes
 /cancer
 /heart
 /kidney
 /liver
 /malaria
 /pneumonia
```

Separate prediction routes are implemented for the individual disease modules.

The application uses:

* Flask for the web server
* Jinja2 HTML templates
* Pickle for traditional ML models
* TensorFlow/Keras for image models
* OpenCV for image processing
* Pillow for image handling
* NumPy and Pandas for data processing

---

# 📂 Project Structure

```text
MEDIFORECAST---Multiple-Disease-Predictor/
│
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
│
├── dataset/
│   ├── Breast_Cancer.csv
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney.csv
│   ├── liver.csv
│   ├── Malaria.zip
│   └── Pneumonia.zip
│
├── models/
│   ├── breast_cancer.pkl
│   ├── diabetes.pkl
│   ├── heart.pkl
│   ├── kidney.pkl
│   ├── liver.pkl
│   ├── malaria.h5
│   └── pneumonia.h5
│
├── notebooks/
│   ├── diabetes.ipynb
│   ├── Breast Cancer.ipynb
│   ├── Heart Disease.ipynb
│   ├── Kidney Disease.ipynb
│   ├── liver.ipynb
│   ├── malaria.ipynb
│   ├── pneumonia.ipynb
│   └── Malaria/
│
├── templates/
│   ├── home.html
│   ├── main.html
│   ├── diabetes.html
│   ├── diabetes_predict.html
│   ├── breast_cancer.html
│   ├── breast_cancer_predict.html
│   ├── heart.html
│   ├── heart_predict.html
│   ├── kidney.html
│   ├── kidney_predict.html
│   ├── liver.html
│   ├── liver_predict.html
│   ├── malaria.html
│   ├── malaria_predict.html
│   ├── pneumonia.html
│   └── pneumonia_predict.html
│
├── static/
│   ├── logo.png
│   ├── logo1.png
│   └── upload.png
│
└── test_images/
    ├── malaria/
    └── pneumonia/
```

---

# 🛠️ Technology Stack

## Programming

* Python

## Web Development

* Flask
* HTML
* Jinja2

## Data Processing

* NumPy
* Pandas

## Machine Learning

* Scikit-learn
* XGBoost

## Deep Learning

* TensorFlow
* Keras
* VGG19
* InceptionResNetV2

## Image Processing

* OpenCV
* Pillow

## Model Serialization

* Pickle
* HDF5/Keras `.h5`

---

# 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/RoshanUpadhyay02/MEDIFORECAST---Multiple-Disease-Predictor.git

cd MEDIFORECAST---Multiple-Disease-Predictor
```

### 2. Install Git LFS

The repository uses **Git LFS** for large datasets and deep-learning model files.

Install Git LFS and then run:

```bash
git lfs install
```

After cloning:

```bash
git lfs pull
```

This is important because files such as the malaria and pneumonia models and image datasets are stored through Git LFS.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

The project's current requirements include:

```text
Flask
numpy
pandas
Pillow
tensorflow
scikit-learn
opencv-python
```

### 4. Run the application

```bash
python app.py
```

The Flask development server will start locally.

Open the address shown in the terminal, typically:

```text
http://127.0.0.1:5000/
```

---

# 🔄 Application Workflow

```text
                 ┌───────────────────┐
                 │   Flask Web App   │
                 └─────────┬─────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
      Structured Data               Medical Image
             │                           │
             ▼                           ▼
     Pre-trained ML Models       Deep Learning Models
             │                           │
             ▼                           ▼
      Disease Prediction          Image Classification
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Prediction Result
```

---

# 📊 Model Summary

| Disease        |  Dataset Size | Input Type        | Saved Model         | Algorithm                     |
| -------------- | ------------: | ----------------- | ------------------- | ----------------------------- |
| Diabetes       |           768 | Tabular           | `diabetes.pkl`      | Gradient Boosting             |
| Breast Cancer  |           569 | Tabular           | `breast_cancer.pkl` | Decision Tree                 |
| Heart Disease  |         1,025 | Tabular           | `heart.pkl`         | Random Forest                 |
| Kidney Disease |           400 | Tabular           | `kidney.pkl`        | Random Forest                 |
| Liver Disease  |           583 | Tabular           | `liver.pkl`         | SVM                           |
| Malaria        | Image dataset | Blood-smear image | `malaria.h5`        | VGG19-based model             |
| Pneumonia      | Image dataset | Chest X-ray       | `pneumonia.h5`      | InceptionResNetV2-based model |

---

# 📈 Reported Notebook Results

The following figures are the results recorded in the included notebooks. They should be interpreted as **notebook/model-development results**, not as a guarantee of real-world clinical performance.

| Disease        | Model             |               Reported Evaluation |
| -------------- | ----------------- | --------------------------------: |
| Diabetes       | Gradient Boosting |          **92.64% test accuracy** |
| Breast Cancer  | Decision Tree     |       **100% recorded accuracy*** |
| Heart Disease  | Random Forest     |          **98.38% test accuracy** |
| Kidney Disease | Random Forest     |          **97.50% test accuracy** |
| Liver Disease  | SVM               |          **71.18% test accuracy** |
| Malaria        | Fine-tuned VGG19  |          **96.04% test accuracy** |
| Pneumonia      | InceptionResNetV2 | **99.04% recorded test accuracy** |

* The breast-cancer notebook evaluates the compared models on the full transformed dataset rather than a held-out test set, so the 100% figure should not be presented as an independent test-set result.

---

# ⚠️ Implementation Notes

The repository contains the original model-training notebooks and the Flask inference application. Some preprocessing performed during model training is not fully reproduced in `app.py`.

For example, several tabular models were trained using `StandardScaler` and/or categorical encoding in their notebooks, while the Flask application directly passes submitted values to the serialized models.

Therefore, the model-development results reported above should **not automatically be interpreted as the accuracy of the current Flask inference implementation**.

The malaria application also resizes uploaded images to `36 × 36` before inference, while the included malaria notebook uses a VGG19 architecture with `128 × 128 × 3` inputs. This should be reviewed before treating the malaria web prediction endpoint as production-ready.

These preprocessing differences are important technical limitations of the current implementation.

---

# 🧪 Test Images

The repository contains sample images for testing:

```text
test_images/
├── malaria/
│   ├── C1_thinF_IMG_20150604_104722_cell_73.png
│   └── C33P1thinF_IMG_20150619_114756a_cell_181.png
│
└── pneumonia/
    ├── CHEST_X_RAY.PNG
    └── CHEST_X_RAY2.PNG
```

---

# 📓 Model Development Notebooks

The repository includes individual notebooks documenting the development and evaluation process:

* `diabetes.ipynb`
* `Breast Cancer.ipynb`
* `Heart Disease.ipynb`
* `Kidney Disease.ipynb`
* `liver.ipynb`
* `malaria.ipynb`
* `pneumonia.ipynb`

These notebooks contain data exploration, preprocessing, model training, evaluation, visualization, model comparison, and model serialization steps.

---

# 🚀 Potential Improvements

Future development could include:

* Consistent preprocessing pipelines between training and Flask inference
* Saving preprocessing objects alongside the trained models
* Using `Pipeline` / `ColumnTransformer` for tabular models
* Correctly matching image preprocessing and input dimensions between training and inference
* Adding probability/confidence outputs
* Adding model explainability
* Improving validation methodology
* Adding automated tests
* Improving error handling
* Adding a production WSGI server
* Containerizing the application
* Deploying the application to a cloud platform
* Adding proper medical-data privacy and security controls
* Adding authentication and patient-history functionality
* Adding automated model/version tracking

---

# ⚠️ Medical Disclaimer

MEDIFORECAST is a machine-learning project intended for **educational and research purposes**.

The predictions generated by this application are not medical diagnoses and should not be used to make healthcare decisions without consultation with a qualified healthcare professional.

The models were developed using publicly available datasets and should not be assumed to generalize to every patient population, clinical setting, imaging device, or geographic region.

---

# 📄 License

This project is licensed under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the complete license text.

---

# 👨‍💻 Author

**Roshan Upadhyay**

GitHub:
https://github.com/RoshanUpadhyay02

---

⭐ If you find this project useful, consider giving the repository a star.
