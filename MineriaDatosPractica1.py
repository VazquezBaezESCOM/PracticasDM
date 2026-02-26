import numpy as np # Manejo de números
import pandas as pd # Dataframes
import seaborn as sns # Visualización de estadísticas
import matplotlib.pyplot as plt # Graficas

from sklearn.model_selection import train_test_split # Dividir datos de entrenamiento-pruena
from sklearn.compose import ColumnTransformer # Modificar columnas
from sklearn.pipeline import Pipeline # Encadenar pasos

from sklearn.preprocessing import OneHotEncoder, StandardScaler 
# OneHotEncoder: convertir categorías 
# StandardScaler: estandarizar valores

from sklearn.impute import SimpleImputer #Rellenar valores faltantes
from sklearn.linear_model import LogisticRegression # Modelo de clasifiación
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Exactitud, matriz dde confusión, precisión de clase

# Análisis de supervivientes del Titanic (probabilidad de supervivencia)

df = sns.load_dataset("titanic")

print("Columnas:", df.columns)
print(df.head())

# Etapa 1: Sample
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "alone"]
target = "survived"

X = df[features]
y = df[target]

# Dividir la muestra

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#Stratify: Manejo por capas de las variables

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train:", y_train.mean(), "Test:", y_test.mean())

# Etapa 2: Explore
print(X_train.dtypes)

print(X_train.isna().sum())

bins = [0, 20, 40, 60, 80, 100]
labels = ["0-20 años", "21-40 años", "41-60 años", "61-80 años", "81-100 años"]
df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)


surv_by_sex = df.groupby("sex")["survived"].mean()
print(surv_by_sex)

# Graficar la probabilidad de supervivencia por sexo
surv_by_sex.plot(kind="bar")
plt.title("Probabilidad de Supervivencia por Sexo")
plt.show()

surv_by_age = df.groupby("age")["survived"].mean()
print(surv_by_age)

# Graficar la probabilidad de supervivencia por edad
surv_by_age.plot(kind="bar")
plt.title("Probabilidad de Supervivencia por Edad")
plt.show()

# Graficar la probabilidad de supervivencia por sexo
surv_by_age_range = df.groupby("age_range", observed=True)["survived"].mean()
print(surv_by_age_range)

surv_by_age_range.plot(kind="bar", color='skyblue', edgecolor='black')
plt.title("Probabilidad de Supervivencia por Rangos de Edad")
plt.show()

# Etapa 3: Modify
numeric_features = ["age", "sibsp", "parch", "fare", "pclass", "alone"]
categorical_features = ["sex", "embarked"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), # Rellenar valores faltantes con la mediana
    ("scaler", StandardScaler()) # Estandarizar los valores
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), # Rellenar valores faltantes con la moda
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Etapa 4: Model

model = LogisticRegression(max_iter=1000)
model1 = DecisionTreeClassifier(random_state=42)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)])

clf1 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model1)])

clf.fit(X_train, y_train)
clf1.fit(X_train, y_train)

# Etapa 5: Assess

y_pred = clf.predict(X_test)
y_pred1 = clf1.predict(X_test)

print("Accuracy Regresión Logística:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Accuracy Árbol de Decisión:", accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

