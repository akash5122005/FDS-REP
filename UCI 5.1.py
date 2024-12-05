import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
df=pd.read_csv("diabetes.csv")
print(df)
print("\nFrequency of Pregnancies:\n")
print(df["Pregnancies"].value_counts())
print("\nFrequency of Glucose:\n")
print(df["Glucose"].value_counts())
print("\nFrequency of BloodPressure:\n")
print(df["BloodPressure"].value_counts())
print("\nFrequency of SkinThickness:\n")
print(df["SkinThickness"].value_counts())
print("\nFrequency of Insulin:\n")
print(df["Insulin"].value_counts())
print("\nFrequency of BMI:\n")
print(df["BMI"].value_counts())
print("\nFrequency of DiabetesPedigreeFunction:\n")
print(df["DiabetesPedigreeFunction"].value_counts())
print("\nFrequency of Age:\n")
print(df["Age"].value_counts())
print("\nFrequency of Outcome:\n")
print(df["Outcome"].value_counts())
print("Mean, Median, Mode, Standard deviation, skewness and Kurtosis\n")
print("Mean of Pregnancies:",df["Pregnancies"].mean())
print("Median of Pregnancies:",df['Pregnancies'].median())
print("Mode of Pregnancies:",df["Pregnancies"].mode())
print("Standard Deviation of Pregnancies:",df["Pregnancies"].std())
print("Skewness of Pregnancies:",df["Pregnancies"].skew())
print("Kurtosis of Pregnancies:",df["Pregnancies"].kurt())
print("\nBivariate Analysis : linear and logistic regression modelling:\n")
x = df['Age']
y = df['BMI']
n = np.size(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean
b1 = Sxy/Sxx
b0 = y_mean-b1*x_mean
print('slope b1 is', b1)
print('intercept b0 is', b0)
y_pred = b1 * x + b0
print("Linear Regression :",y_pred.mean())
#LOGISTIC REgression
X = df[['Age', 'Pregnancies']]
y = df['Outcome']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Creating a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# Making predictions on the test data
y_pred = model.predict(X_test_scaled)
# Evaluating the model
accuracy = model.score(X_test_scaled, y_test)
print("\nLogistic Regression:\n")
print(f"Accuracy: {accuracy:.2f}")
print("\nMultiple Regression:\n")
model1 = sm.OLS.from_formula(' Pregnancies ~ Age+ Outcome ', df).fit()
print(model1.summary())
