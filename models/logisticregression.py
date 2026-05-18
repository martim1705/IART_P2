import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

import os


# ler dataset
df = pd.read_csv("data/dataset1.csv")

print("\nDataset carregado:")
print(df.head())


# inputs (features)
X = df.drop(columns=["nome", "risco_lesao"])

# resultado esperado
y = df["risco_lesao"]


# dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# pipeline: normaliza features + regressao logistica
# (regressao logistica precisa de features na mesma escala)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])


# treinar
model.fit(X_train, y_train)

print("\nModelo treinado.")


# fazer previsoes
pred = model.predict(X_test)


# precisao
print("\nAccuracy:")
print(accuracy_score(y_test, pred))


# metricas detalhadas
print("\nClassification report:")
print(classification_report(y_test, pred))


# matriz confusao
print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))


# coeficientes (equivalente a importancia das variaveis na reg. logistica)
print("\nCoeficientes por classe:")

clf = model.named_steps["clf"]

for classe, coefs in zip(clf.classes_, clf.coef_):
    print(f"\nClasse {classe}:")
    for coluna, coef in zip(X.columns, coefs):
        print(f"  {coluna}: {coef:+.3f}")


# guardar modelo
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/injury_model_logisticregression.pkl"
)

print("\nModelo guardado.")


'''

1. Le uma tabela com jogadores
2. Separa os dados em:
   - caracteristicas dos jogadores (X)
   - risco de lesao (y)
3. Da parte dos dados ao algoritmo para ele aprender
4. Guarda outra parte para testar
5. Normaliza as features (StandardScaler) - importante para reg. logistica
6. Treina a regressao logistica multinomial (3 classes de risco)
7. Pede ao modelo para prever o risco
8. Compara as previsoes com as respostas verdadeiras
9. Mostra accuracy, classification report e matriz de confusao
10. Mostra os coeficientes de cada feature por classe
11. Guarda o modelo para usar depois na app

'''
