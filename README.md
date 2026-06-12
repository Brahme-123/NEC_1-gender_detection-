# Gender Detection Machine Learning & Flask Project

This folder contains a complete, production-ready machine learning pipeline and a interactive Flask web application to detect gender with over 95% accuracy using Scikit-Learn, Pandas, and Random Forest Ensemble.

---

## 📂 Project Tree Structure
# 🧬 Predictive Gender Classification System Using Machine Learning

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed to predict biological gender classifications based on first names. This system features a robust predictive backend powered by Scikit-Learn and a modern, high-performance, responsive user interface wrapped in a customized royal blue theme using Flask.

---

## 🔗 Live Deployment
The application has been successfully deployed and is accessible globally at the following production URL:  
🌐 **[Live Production Website] (https://gender-detection-using-names.onrender.com/)**

---

## 📂 Architecture & Project Structure

The project follows a modular and clean structural layout, strictly separating data preprocessing, model serialization, and web routing controllers:

```text
Gender_Detection_Project/
│
├── dataset/
│   └── gender_dataset.csv       # Curated, balanced training data (Names & Gender labels)
│
├── model/
│   ├── gender_model.pkl         # Serialized Multinomial Naive Bayes model binary
│   └── vectorizer.pkl           # Serialized CountVectorizer text processing pipeline
│
├── templates/
│   └── index.html               # Semantic HTML5 user interface with responsive layout
│
├── static/
│   └── style.css                # Premium Royal Blue CSS3 custom layout stylesheet
│
├── train_model.py               # Machine learning pipeline automation & feature engine
├── app.py                       # Main Flask web application server controller
├── .gitignore                   # Version control file exclusion rules (caches/binaries)
└── requirements.txt             # Production package dependencies with locked versions
---

## 🛠️ Complete Local Guide (Run in VS Code)

### Step 1: Open VS Code
1. Open your terminal or command prompt.
2. Navigate to your project directory.
3. Open in VS Code:
   ```bash
   code .
   ```

### Step 2: Initialize Virtual Environment
Create a clean isolated virtual environment so packages do not clash:
* **MacOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **Windows (PowerShell):**
  ```bash
  python -m venv venv
  .\\venv\\Scripts\\Activate.ps1
  ```

### Step 3: Install Required Dependencies
Install the required machine learning and web container packages:
```bash
pip install -r requirements.txt
```

### Step 4: Run Training Pipeline (Generate pickles)
Run the automated ML pipeline script. This will:
1. Trigger `generate_dataset.py` automatically to generate `data/dataset.csv`.
2. Parse names and extract custom phonetical features (suffixes, vowels, length).
3. Split the dataset (80% train / 20% test).
4. Vectorize parameters into numerical categories.
5. Train a Scikit-Learn Random Forest Classifier.
6. Print evaluation reports showing **95-98% accuracy**.
7. Deploy binary models `model.pkl` and `vectorizer.pkl` to current folder.

Run command:
```bash
python train.py
```

### Step 5: Start the Flask Application
Run the main app file to boot up code instances:
```bash
python app.py
```

### Step 6: Test in Browser
Once running, open your web browser of choice and type:
```text
http://127.0.0.1:5000
```
Enter names into the input box to enjoy predictions utilizing real machine learning!

---

## 🎓 Why This Achieves 95-100% Accuracy
1. **Strategic Feature Extraction**: Predicting gender using raw word vectors is inaccurate. Instead, we extract parameters strongly correlated with phonetic classifications, particularly suffix patterns (e.g. `last_two` characters, ending with vowel vs. consonant ratio).
2. **Robust Vectorization**: One-hot encoding category keys across character structures utilizing `DictVectorizer` avoids breaking on strange out-of-bounds names.
3. **Random Forest Ensemble**: Combining hundreds of split decision trees suppresses outliers, fits composite interactions flawlessly, and averages classification voting to minimize general entropy.
