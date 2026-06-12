import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def train_gender_model():
    # Ensure folders exist
    os.makedirs('model', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    dataset_path = "dataset/gender_dataset.csv"
    
    # Text data directly inside Python
    raw_data = """name,gender
Rahul,Male
Ramesh,Male
Suresh,Male
Mahesh,Male
Arjun,Male
Kiran,Male
Vijay,Male
Ajay,Male
Ravi,Male
Praveen,Male
Rohit,Male
Karthik,Male
Manoj,Male
Teja,Male
Naveen,Male
Krishna,Male
Sai,Male
Rajesh,Male
Deepak,Male
Anil,Male
Lakshmi,Female
Priya,Female
Divya,Female
Anusha,Female
Keerthana,Female
Sravani,Female
Bhavya,Female
Kavya,Female
Pooja,Female
Sneha,Female
Swathi,Female
Harika,Female
Nandini,Female
Sowmya,Female
Deepika,Female
Meghana,Female
Anjali,Female
Aishwarya,Female
Sirisha,Female
Durga,Female"""

    # Overwrite the file with fresh data to avoid empty file error
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(raw_data.strip())
        
    print("Dataset created and checked successfully!")
        
    # Read the freshly written file
    df = pd.read_csv(dataset_path)
    
    # Features & Labels
    X = df["name"].astype(str)
    y = df["gender"]
    
    # Vectorization
    cv = CountVectorizer()
    X_vectorized = cv.fit_transform(X)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )
    
    # Initialize & Train Model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully with Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save Model and Vectorizer pipelines
    with open("model/gender_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
        
    with open("model/vectorizer.pkl", "wb") as cv_file:
        pickle.dump(cv, cv_file)
        
    print("All ML files saved successfully in 'model/' directory.")

if __name__ == "__main__":
    train_gender_model()