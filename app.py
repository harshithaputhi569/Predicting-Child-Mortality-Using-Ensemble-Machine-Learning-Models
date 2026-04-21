from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model (ensemble)
model_data = joblib.load("model/best_model.pkl")
models = model_data['models']
weights = model_data['weights']
best_threshold = model_data['best_threshold']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict_v2.html")

@app.route("/result", methods=["POST"])
def result():
    # Extract all 10 required features in the correct order
    # Debug: Print all form keys to console
    received_keys = list(request.form.keys())
    print("Received Keys:", received_keys)

    try:
        # Accessing with brackets triggers BadRequestKeyError if missing
        birth_weight = float(request.form["birth_weight"])
        mother_education_no = float(request.form["mother_education_no"])
        wealth_index_poor = float(request.form["wealth_index_poor"])
        institutional_delivery_yes = float(request.form["institutional_delivery_yes"])
        institutional_delivery_no = float(request.form["institutional_delivery_no"])
        vaccination_yes = float(request.form["vaccination_yes"])
        vaccination_no = float(request.form["vaccination_no"])
        father_age = float(request.form["father_age"])
        mother_age = float(request.form["mother_age"])
        antenatal_visits = float(request.form["antenatal_visits"])
    except Exception as e:
        # Catch all to ensure we see what's wrong
        return (
            f"<h3>Error: Missing or Invalid Data</h3>"
            f"<p><b>Error Details:</b> {str(e)}</p>"
            f"<p><b>Received Keys:</b> {received_keys}</p>"
            f"<p><b>Expected Keys:</b> ['birth_weight', 'mother_education_no', ...]</p>"
            f"<br><a href='/predict'>Go Back</a>"
        ), 400

    # Create feature array in the exact order expected by the model
    features = np.array([[birth_weight, mother_education_no, wealth_index_poor, 
                          institutional_delivery_yes, institutional_delivery_no,
                          vaccination_yes, vaccination_no, father_age, mother_age, 
                          antenatal_visits]])
    
    # Make weighted ensemble predictions
    prediction_list = []
    weight_list = []
    for name, model in models.items():
        prob = model.predict_proba(features)[0][1]
        prediction_list.append(prob)
        weight_list.append(weights[name])

    risk_score = round(np.average(prediction_list, weights=weight_list) * 100, 2)

    if risk_score >= 50:
        level = "High Risk"
    else:
        level = "Low Risk"
        
    # Calculate feature contributions (Importance)
    feature_names = [
        'Birth Weight', 'Mother Education', 'Wealth Index (Poor)', 
        'Institutional Delivery (Yes)', 'Institutional Delivery (No)',
        'Vaccination (Yes)', 'Vaccination (No)', 'Father Age', 'Mother Age', 
        'Antenatal Visits'
    ]
    
    # Aggregate feature importances from all models in the ensemble
    # Initialize zero array
    final_importances = np.zeros(len(feature_names))
    total_weight = sum(weights.values())
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # Weighted importance
            model_weight = weights[name]
            final_importances += model.feature_importances_ * model_weight
            
    # Normalize
    final_importances /= total_weight
    
    # Pair feature names with their importance scores
    importance_list = sorted(zip(feature_names, final_importances), key=lambda x: x[1], reverse=True)
    
    # Get top 3 factors
    top_factors = [{'name': name, 'score': round(score * 100, 1)} for name, score in importance_list[:3]]

    return render_template(
        "result.html",
        risk=risk_score,
        level=level,
        factors=top_factors
    )

if __name__ == "__main__":
    app.run(debug=True)