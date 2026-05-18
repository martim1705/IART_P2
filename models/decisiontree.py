import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import os

df = pd.read_csv("data/dataset1.csv")

X = df.drop(columns=["nome", "risco_lesao"])
y = df["risco_lesao"]

model = DecisionTreeClassifier(
    random_state=40,
    max_depth=7
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=40,
    stratify=y
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification report:")
print(classification_report(y_test, pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))

# Guardar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/dt_injury_model.pkl")

print("\nModelo guardado.")