import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ler dataset
df = pd.read_csv("data/dataset1.csv")

print("\nDataset carregado:")
print(df.head())


# inputs (features)
X = df.drop(
    columns=["nome", "risco_lesao"]
)

# resultado esperado
y = df["risco_lesao"]


# dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# criar modelo
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# treinar
model.fit(
    X_train,
    y_train
)

print("\nModelo treinado.")


# fazer previsões
pred = model.predict(
    X_test
)


# precisao
acc = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy:")
print(acc)


# métricas detalhadas
print("\nClassification report:")
print(
    classification_report(
        y_test,
        pred
    )
)


# matriz confusão
print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        pred
    )
)


# importância atributos
print("\nImportância das variáveis:")

for coluna, importancia in zip(
        X.columns,
        model.feature_importances_
):

    print(
        f"{coluna}: {importancia:.3f}"
    )


# guardar modelo
joblib.dump(model, "models/rf_injury_model.pkl")

print(
    "\nModelo guardado."
)