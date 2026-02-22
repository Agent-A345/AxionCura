# Stroke Prediction ML Model

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from app import model

# 1. Load data
df = pd.read_csv("stroke.csv")
print("Data shape:", df.shape)
print(df.head())

# 2. Quick EDA
print("\nTarget distribution:")
print(df['stroke'].value_counts())

# 3. Features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 4. Preprocessing:
#    - numeric: age, avg_glucose_level, bmi
#    - binary/numeric already: hypertension, heart_disease
#    - categorical: ever_married, smoking_status, gender, work_type, residence_type

numeric_features = ["age", "avg_glucose_level", "bmi"]
numeric_transformer = StandardScaler()

categorical_features = ["ever_married", "smoking_status", "gender", "work_type", "residence_type"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"  # keep hypertension, heart_disease as-is
)

# 5. Build pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42))
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Train model
pipeline.fit(X_train, y_train)
print("\nModel trained.")

import joblib
joblib.dump(model, "stroke_pipeline.joblib", compress=3)

# 8. Predict & evaluate
y_pred = pipeline.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Example: predict for a new patient
new_patient = pd.DataFrame({
    "age": [67],
    "hypertension": [1],
    "heart_disease": [0],
    "ever_married": ["Yes"],
    "avg_glucose_level": [160.0],
    "bmi": [31.2],
    "smoking_status": ["formerly smoked"],
    "gender": ["Male"],
    "work_type": ["Private"],
    "residence_type": ["Urban"]
})

pred = pipeline.predict(new_patient)[0]
print("\n New patient prediction (1=stroke, 0=no):", pred)
