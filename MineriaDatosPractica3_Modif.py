import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.tree import DecisionTreeClassifier

#1. Cargar datos
#Data: Breast Cancer Wisconsin
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print("Primeras filas del dataset:")
print(X.head())

print("\nDistribución de clases:")
print(y.value_counts())

print("\nNombre de las clases:")
print(data.target_names)

#2. Divivir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", y_train.shape)

#3. Crear los modelos

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

modelo_arbol = DecisionTreeClassifier(max_depth = 4, random_state=42)

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)
modelo_arbol.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)
y_pred_tree = modelo_arbol.predict(X_test)

# Evaluación de las métricas del modelo
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

acc_tree = accuracy_score(y_test, y_pred_tree)
prec_tree = precision_score(y_test, y_pred_tree)
rec_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)
cm_tree = confusion_matrix(y_test, y_pred_tree)

print("\n===== MODELO: REGRESIÓN LOGÍSTICA =====")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)

print("Matriz de confusión:")
print(cm)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv = cv,
    scoring="f1"
)

print("\n===== VALIDACIÓN CRUZADA =====")
print ("F1-score por fold:", cv_scores)
print("Promedio F1-score:", cv_scores.mean())
print("Desviación estándar", cv_scores.std())

plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), cv_scores, marker='o')
plt.title("F1-score por fold en validación cruzada - Regresión Lineal")
plt.xlabel("Fold")
plt.ylabel("F1-score")
plt.grid(True)
plt.show()

print("\n===== MODELO: ÁRBOL DE DECISIÓN =====")
print("Accuracy:", acc_tree)
print("Precision:", prec_tree)
print("Recall:", rec_tree)
print("F1-score:", f1_tree)

print("Matriz de confusión:")
print(cm)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_tree, target_names=data.target_names))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    modelo_arbol,
    X,
    y,
    cv = cv,
    scoring="f1"
)

print("\n===== VALIDACIÓN CRUZADA =====")
print ("F1-score por fold:", cv_scores)
print("Promedio F1-score:", cv_scores.mean())
print("Desviación estándar", cv_scores.std())

plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), cv_scores, marker='o')
plt.title("F1-score por fold en validación cruzada - Árbol de Decisión")
plt.xlabel("Fold")
plt.ylabel("F1-score")
plt.grid(True)
plt.show()

print("\n=====CONCLUSIÓN=====")
print("Ambos modelos son útiles para clasificar las instancias del conjunto de datos.")
print("Sin embargo, para este caso especifico, el mejor modelo de clasificación es  \nel que utiliza la regresión lineal por la cantidad de iteraciones que tiene.")
print("De igual manera, sus predicciones son de mayor calidad de acuerdo a \nlas métricas con las que evaluamos el modelo.")