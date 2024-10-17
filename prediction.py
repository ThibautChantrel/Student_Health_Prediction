import time
import threading
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global variables for models and scaler
stress_model = None
gpa_model = None
scaler_stress = None
scaler_gpa = None

# Simulating data loading for Sleep Health
def train_models():
    global stress_model, gpa_model, scaler_stress, scaler_gpa

    # Model for Stress Level Prediction
    fileName = "Modified_sleep_health_and_lifestyle_dataset.csv"
    FullPath = "./Modified_csv/" + fileName
    sleep_health_data = pd.read_csv(FullPath, sep=';')  # Adjust the separator as needed
    X_sleep = sleep_health_data[['Gender', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Systolic']]
    y_sleep = sleep_health_data['Stress Level']

    # Model for GPA Prediction
    fileName = "Student_performance_data_V2.csv"
    FullPath = "./Modified_csv/" + fileName
    performance_data = pd.read_csv(FullPath, sep=';')  # Adjust the separator as needed
    X_gpa = performance_data[['Age', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports']]
    y_gpa = performance_data['GPA']
    
    # Split the datasets for training
    X_sleep_train, X_sleep_test, y_sleep_train, y_sleep_test = train_test_split(X_sleep, y_sleep, test_size=0.2, random_state=42)
    X_gpa_train, X_gpa_test, y_gpa_train, y_gpa_test = train_test_split(X_gpa, y_gpa, test_size=0.2, random_state=42)

    # Standardize the features
    scaler_stress = StandardScaler()
    scaler_gpa = StandardScaler()

    X_sleep_train_scaled = scaler_stress.fit_transform(X_sleep_train)
    X_gpa_train_scaled = scaler_gpa.fit_transform(X_gpa_train)

    # Train Gradient Boosting Regressor for Stress Level
    stress_model = GradientBoostingRegressor()
    stress_model.fit(X_sleep_train_scaled, y_sleep_train)

    # Train Linear Regression for GPA Prediction
    gpa_model = LinearRegression()
    gpa_model.fit(X_gpa_train_scaled, y_gpa_train)

@app.route('/')
def index():
    return render_template('index.html')  # Page d'accueil

@app.route('/loading')
def loading_page():
    return render_template('loading.html')  # HTML page that shows loading

@app.route('/description')
def description():
    return render_template('description.html')  # Description du projet

@app.route('/data')
def data():
    # Génération de graphiques explicatifs
    fileName = "Modified_sleep_health_and_lifestyle_dataset.csv"
    FullPath = "./Modified_csv/" + fileName
    sleep_health_data = pd.read_csv(FullPath, sep=';')

    # Graphique 1: Histogramme des niveaux de stress
    plt.figure(figsize=(8, 5))
    sleep_health_data['Stress Level'].hist(bins=30, edgecolor='black')
    plt.title('Distribution des niveaux de stress')
    plt.xlabel('Niveau de stress')
    plt.ylabel('Fréquence')

    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot1_url = base64.b64encode(img1.getvalue()).decode()

    plt.clf()  # Effacer le graphique précédent

    # Graphique 2: Relation entre la durée du sommeil et le niveau de stress
    plt.figure(figsize=(8, 5))
    plt.scatter(sleep_health_data['Sleep Duration'], sleep_health_data['Stress Level'], alpha=0.5)
    plt.title('Durée du sommeil vs Niveau de stress')
    plt.xlabel('Durée du sommeil (heures)')
    plt.ylabel('Niveau de stress')

    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot2_url = base64.b64encode(img2.getvalue()).decode()

    return render_template('data.html', plot1_url=plot1_url, plot2_url=plot2_url)

@app.route('/form')
def form_page():
    return render_template('form.html')  # Form to input features

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    gender = int(request.form['gender'])
    sleep_duration = float(request.form['sleep_duration'])
    quality_sleep = float(request.form['quality_sleep'])
    physical_activity = int(request.form['physical_activity'])
    heart_rate = int(request.form['heart_rate'])
    daily_steps = int(request.form['daily_steps'])
    systolic = int(request.form['systolic'])

    age = int(request.form['age'])
    study_time_weekly = float(request.form['study_time_weekly'])
    absences = int(request.form['absences'])
    tutoring = int(request.form['tutoring'])
    parental_support = int(request.form['parental_support'])
    extracurricular = int(request.form['extracurricular'])
    sports = int(request.form['sports'])

    # Make prediction for stress
    stress_features = np.array([[gender, sleep_duration, quality_sleep, physical_activity, heart_rate, daily_steps, systolic]])
    stress_features_scaled = scaler_stress.transform(stress_features)
    stress_prediction = stress_model.predict(stress_features_scaled)[0]

    # Make prediction for GPA
    gpa_features = np.array([[age, study_time_weekly, absences, tutoring, parental_support, extracurricular, sports]])
    gpa_features_scaled = scaler_gpa.transform(gpa_features)
    gpa_prediction = gpa_model.predict(gpa_features_scaled)[0]

    # Show predictions
    return render_template('result.html', stress=stress_prediction, gpa=gpa_prediction)

def start_training():
    train_models()

# Start model training in the background thread
training_thread = threading.Thread(target=start_training)
training_thread.start()

if __name__ == "__main__":
    app.run(debug=True)
