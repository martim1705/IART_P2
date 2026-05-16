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
    random_state=42,
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

# guardar modelo

os.makedirs("models", exist_ok=True)
joblib.dump(
    model,
    "models/injury_model_decisiontree.pkl"
)

print(
    "\nModelo guardado."
)

'''

1. Lê uma tabela com jogadores - 
2. Separa os dados em:
   - características dos jogadores
   - risco de lesão
3. Dá parte dos dados ao algoritmo para ele aprender
4. Guarda outra parte para testar
5. Treina a árvore de decisão
6. Pede à árvore para prever o risco
7. Compara as previsões com as respostas verdadeiras
8. Mostra se o modelo acertou muito ou pouco
9. Guarda o modelo para usar depois na app

'''