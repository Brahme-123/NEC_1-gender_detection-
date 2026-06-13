import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)


EXACT_NAMES_DB = {
    # Boys Names (Male)
    "sai": "Male", "teja": "Male", "krishna": "Male", "surya": "Male", "vamsi": "Male",
    "siva": "Male", "rahul": "Male", "ramesh": "Male", "suresh": "Male", "mahesh": "Male",
    "arjun": "Male", "kiran": "Male", "vijay": "Male", "ajay": "Male", "ravi": "Male",
    "praveen": "Male", "rohit": "Male", "karthik": "Male", "manoj": "Male", "naveen": "Male",
    "rajesh": "Male", "deepak": "Male", "anil": "Male", "vikram": "Male", "charan": "Male",
    "pawan": "Male", "kalyan": "Male", "ram": "Male", "ganesh": "Male", "anand": "Male",
    "harish": "Male", "sandeep": "Male", "satish": "Male", "ntr": "Male", "prabhas": "Male",
    "venkat": "Male", "prasad": "Male", "naresh": "Male", "srikanth": "Male", "gopi": "Male",
    "srinu": "Male", "babu": "Male", "mohan": "Male", "koti": "Male", "chari": "Male",

    # Girls Names (Female)
    "lakshmi": "Female", "priya": "Female", "divya": "Female", "anusha": "Female", 
    "keerthana": "Female", "sravani": "Female", "bhavya": "Female", "kavya": "Female", 
    "pooja": "Female", "sneha": "Female", "swathi": "Female", "harika": "Female", 
    "nandini": "Female", "sowmya": "Female", "deepika": "Female", "meghana": "Female", 
    "anjali": "Female", "aishwarya": "Female", "sirisha": "Female", "durga": "Female",
    "radha": "Female", "sita": "Female", "geetha": "Female", "vani": "Female", 
    "kalyani": "Female", "madhuri": "Female", "rani": "Female", "jyothi": "Female", 
    "roja": "Female", "uma": "Female", "sarada": "Female", "hema": "Female",
    "shanti": "Female", "lavanya": "Female", "ramya": "Female", "supriya": "Female"
}

# Load ML Model and Vectorizer safely
model = None
cv = None

if os.path.exists("model/gender_model.pkl") and os.path.exists("model/vectorizer.pkl"):
    with open("model/gender_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    input_name = ""

    if request.method == "POST":
        input_name = request.form.get("name", "").strip()
        
        if input_name:
            name_lower = input_name.lower()
            
            if name_lower in EXACT_NAMES_DB:
                prediction = EXACT_NAMES_DB[name_lower]
                confidence = "100.0"
            elif model and cv:
                vect_name = cv.transform([input_name])
                pred = model.predict(vect_name)[0]
                prob = model.predict_proba(vect_name)[0]
                
                prediction = pred
                max_prob = max(prob) * 100
                confidence = f"{max_prob:.1f}"
            else:
                prediction = "Model Not Trained"
                confidence = "0.0"

    
    return render_template("index.html", name=input_name, prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)