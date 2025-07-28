from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask_sqlalchemy import SQLAlchemy
import tempfile

app = Flask(__name__)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test_results.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Define the TestResult model
class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    bp = db.Column(db.Float, nullable=False)
    skin_thickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    dpf = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    exercise = db.Column(db.Integer, nullable=False)
    family_history = db.Column(db.Integer, nullable=False)
    diet = db.Column(db.Integer, nullable=False)
    homa_ir = db.Column(db.Float, nullable=False)
    estimated_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    oxygen_level = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    stress_index = db.Column(db.Integer, nullable=False)
    nbmi = db.Column(db.Float, nullable=False)
    adjusted_insulin = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)

# Create database tables (if they don't exist)
with app.app_context():
    db.create_all()

# Load the trained model and scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Function to calculate HOMA-IR (Insulin Resistance Index)
def calculate_homa_ir(glucose, insulin):
    return round((glucose * insulin) / 405, 2) if insulin > 0 else 0

# Estimation Functions for Additional Health Parameters
def estimate_bp(age, bmi, stress_level):
    return round(120 + (bmi - 25) * 0.5 + (stress_level * 2) - (age * 0.1), 2)

def estimate_heart_rate(exercise, age):
    return round(70 - (exercise * 2) + (age * 0.3), 2)

def estimate_oxygen_level(bmi, breathing_rate):
    return round(98 - (bmi * 0.2) - (breathing_rate * 0.5), 2)

def estimate_cholesterol(diet, bmi, family_history):
    return round(180 + (bmi - 25) * 2 + (10 * family_history) - (5 * diet), 2)

# New Functions for Additional Calculations
def calculate_stress_index(stress_level):
    # For demonstration, simply return the provided stress level.
    return stress_level

def calculate_nbmi(bmi):
    # Normalized BMI based on an ideal BMI value (e.g., 22.5)
    return round(bmi / 22.5, 2)

def calculate_adjusted_insulin(insulin, nbmi):
    return round(insulin * nbmi, 2)

# New function to generate health suggestions based on stress and insulin values
def get_health_suggestions(stress_index, insulin):
    suggestions = []
    # Thresholds can be adjusted as needed
    if stress_index > 7:
        suggestions.append("High stress detected. Consider stress management techniques such as meditation, yoga, and deep breathing exercises. Ensure adequate sleep and try to reduce workload.")
    if insulin > 25:
        suggestions.append("High insulin levels detected. Reduce intake of refined sugars and carbohydrates, increase dietary fiber and lean protein, and consult a healthcare professional for personalized advice.")
    return suggestions

# Function to get reasons for prediction
def get_reasons(features, prediction):
    reasons = []
    if features[1] > 140:
        reasons.append("High glucose level detected.")
    if features[2] < 60:
        reasons.append("Low blood pressure observed.")
    if features[5] > 30:
        reasons.append("High BMI indicates obesity risk.")
    if features[9] == 1:
        reasons.append("Family history of diabetes detected.")
    if features[10] == 1:
        reasons.append("Poor diet detected.")
    if prediction == "Diabetic" and not reasons:
        reasons.append("Multiple risk factors suggest diabetes.")
    elif prediction == "Not Diabetic" and not reasons:
        reasons.append("No significant risk factors detected.")
    return reasons

# Function to generate prediction and graph
def predict_diabetes(features):
    features_array = np.array([features]).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    reasons = get_reasons(features, result)
    image_path = "images/diabetics.webp" if result == "Diabetic" else "images/not-diabetic.webp"
    graph_url = generate_graph(features)
    return result, reasons, image_path, graph_url

# Function to generate graph from features
def generate_graph(features):
    plt.figure(figsize=(6, 4))
    labels = ["Pregnancies", "Glucose", "BP", "Skin Thickness", "Insulin", "BMI",
              "DPF", "Age", "Exercise", "Family History", "Diet"]
    plt.barh(labels, features, color=['blue', 'red', 'green', 'orange', 'purple',
                                       'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow'])
    plt.xlabel("Values")
    plt.title("User Input Features Visualization")
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{graph_url}"

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Instructions Page
@app.route("/instructions")
def instructions():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get user input
        pregnancies = int(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        bp = float(request.form["bp"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])  # Diabetes Pedigree Function
        age = int(request.form["age"])
        exercise = int(request.form["exercise"])
        family_history = int(request.form["family_history"])
        diet = int(request.form["diet"])
        stress_level = int(request.form.get("stress_level", 0))
        breathing_rate = int(request.form.get("breathing_rate", 0))

        # Prepare feature array for model prediction (using original BP input)
        features = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age, exercise, family_history, diet]

        # Calculate additional health parameters
        homa_ir = calculate_homa_ir(glucose, insulin)
        estimated_bp = estimate_bp(age, bmi, stress_level)
        heart_rate = estimate_heart_rate(exercise, age)
        oxygen_level = estimate_oxygen_level(bmi, breathing_rate)
        cholesterol = estimate_cholesterol(diet, bmi, family_history)
        stress_index = calculate_stress_index(stress_level)
        nbmi = calculate_nbmi(bmi)
        adjusted_insulin = calculate_adjusted_insulin(insulin, nbmi)

        # Predict diabetes using the features array
        prediction_result, reasons, image_path, graph_url = predict_diabetes(features)

        # Get health suggestions if stress and/or insulin are high
        suggestions = get_health_suggestions(stress_index, insulin)

        # Create a new TestResult record
        test_record = TestResult(
            pregnancies=pregnancies,
            glucose=glucose,
            bp=bp,
            skin_thickness=skin_thickness,
            insulin=insulin,
            bmi=bmi,
            dpf=dpf,
            age=age,
            exercise=exercise,
            family_history=family_history,
            diet=diet,
            homa_ir=homa_ir,
            estimated_bp=estimated_bp,
            heart_rate=heart_rate,
            oxygen_level=oxygen_level,
            cholesterol=cholesterol,
            stress_index=stress_index,
            nbmi=nbmi,
            adjusted_insulin=adjusted_insulin,
            prediction=prediction_result
        )
        db.session.add(test_record)
        db.session.commit()

        return render_template("result.html",
                               prediction=prediction_result,
                               reasons=reasons,
                               suggestions=suggestions,
                               image_path=image_path,
                               graph_url=graph_url,
                               homa_ir=homa_ir,
                               estimated_bp=estimated_bp,
                               heart_rate=heart_rate,
                               oxygen_level=oxygen_level,
                               cholesterol=cholesterol,
                               stress_index=stress_index,
                               nbmi=nbmi,
                               adjusted_insulin=adjusted_insulin)
    return render_template("predict.html")

# Route to view all test results from the SQLite database
@app.route("/results")
def view_results():
    records = TestResult.query.all()
    if records:
        return render_template("results.html", results=records)
    return "No test results found. Please run a prediction first."

# Route to download a PDF report for a specific record by its database ID
@app.route("/download/<int:record_id>")
def download(record_id):
    record = TestResult.query.get(record_id)
    if record:
        return download_pdf(record)
    return "Record not found."

# Function to generate a PDF report from a record (using ReportLab)
def download_pdf(record):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    c.drawString(50, 750, "Diabetes Test Report")
    c.drawString(50, 730, f"Pregnancies: {record.pregnancies}")
    c.drawString(50, 710, f"Glucose: {record.glucose}")
    c.drawString(50, 690, f"BP: {record.bp}")
    c.drawString(50, 670, f"Skin Thickness: {record.skin_thickness}")
    c.drawString(50, 650, f"Insulin: {record.insulin}")
    c.drawString(50, 630, f"BMI: {record.bmi}")
    c.drawString(50, 610, f"Diabetes Pedigree Function: {record.dpf}")
    c.drawString(50, 590, f"Age: {record.age}")
    c.drawString(50, 570, f"Exercise: {record.exercise}")
    c.drawString(50, 550, f"Family History: {record.family_history}")
    c.drawString(50, 530, f"Diet: {record.diet}")
    c.drawString(50, 510, f"HOMA-IR: {record.homa_ir}")
    c.drawString(50, 490, f"Estimated BP: {record.estimated_bp}")
    c.drawString(50, 470, f"Heart Rate: {record.heart_rate}")
    c.drawString(50, 450, f"Oxygen Level: {record.oxygen_level}")
    c.drawString(50, 430, f"Cholesterol: {record.cholesterol}")
    c.drawString(50, 410, f"Stress Index: {record.stress_index}")
    c.drawString(50, 390, f"Normalized BMI: {record.nbmi}")
    c.drawString(50, 370, f"Adjusted Insulin: {record.adjusted_insulin}")
    c.drawString(50, 350, f"Prediction: {record.prediction}")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="test_report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)
 