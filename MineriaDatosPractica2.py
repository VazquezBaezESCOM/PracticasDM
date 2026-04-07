import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#1. Carga de datos a utilizar (Telco Costumer Churn)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Visualización inicial de los datos:")
print(df.head())
#df.info()

características = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
objetivo = "Churn"
X = df[características]
y = df[objetivo]
#División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train:", y_train.mean(), "Test:", y_test.mean())

contract_churn = df.groupby("Contract")["Churn"].mean()
print(contract_churn)

payment_churn = df.groupby("PaymentMethod")["Churn"].mean()
print(payment_churn)

contract_churn.plot(kind="bar")
plt.title("Tipos de contrato vs Churn")
plt.xlabel("Tipo de contrato")
plt.ylabel("Tasa de churn")
plt.show()

payment_churn.plot(kind="bar")
plt.title("Método de pago vs Churn")
plt.xlabel("Método de pago")
plt.ylabel("Tasa de churn")
plt.show()