import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Safely load ML artifacts
MODEL_PATH = "model/gender_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    model, vectorizer = None, None
    print("⚠️ WARNING: Model files not found. Please run 'python train_model.py' first!")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return render_template("index.html", error="Model is not trained yet. Run train_model.py.")
        
    name_input = request.form.get("name", "").strip()
    if not name_input:
        return render_template("index.html", error="Please enter a valid name.")
    
    # Transform input data and predict
    vectorized_data = vectorizer.transform([name_input])
    prediction = model.predict(vectorized_data)[0]
    
    # Optional: Calculate probability confidence
    probabilities = model.predict_proba(vectorized_data)[0]
    classes = model.classes_
    confidence = max(probabilities) * 100

    return render_template(
        "index.html", 
        prediction=prediction, 
        name=name_input, 
        confidence=f"{confidence:.1f}%"
    )

if __name__ == "__main__":
    app.run(debug=True)