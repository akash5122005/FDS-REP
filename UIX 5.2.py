df1=pd.read_csv("diabetes_data_upload.csv")
print(df1)
print()
#Univariate analysis
#---> frequency of age
print("\nFrequency of Age:")
print(df1["Age"].value_counts())
#---> frequency of gender
print("\nFrequency of Gender:")
print(df1["Gender"].value_counts())
#---> frequencey of polyuria
print("\nFrequency of Polyuria:")
print(df1["Polyuria"].value_counts())
#---> frequency of delayed healing
print("\nFrequency of delayed healing:")
print(df1["delayed healing"].value_counts)
#--->frequency of class
print("\nFrequency of class:")
print(df1["class"].value_counts())
#---> Mean,median,mode,standard deviation,skewness,kurtosis
print("\nMean,median,mode,standard deviation,skewness,kurtosis:")
print("Mean of Age:",df1["Age"].mean())
print("Median of Age:",df1['Age'].median())
print("Mode of Age:",df1["Age"].mode())
print("Standard Deviation of Age:",df1["Age"].std())
print("Skewness of Age:",df1["Age"].skew())
print("Kurtosis of Age:",df1["Age"].kurt())
#---> Birvariate analysis:Linear and Logistic regression modelling
print("\nLinear Regression:")
x = df1['Age']
y = df1['Polyuria']
n = np.size(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean
b1 = Sxy/Sxx
b0 = y_mean-b1*x_mean
print('slope b1 is:', b1)
print('intercept b0 is:', b0)
y_pred = b1 * x + b0
print("Linear Regression :",y_pred.mean())
print("\nLogistic regression:")
X = df1[['Age', 'weakness']]
y = df1['Polyuria']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
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
print(f"Accuracy: {accuracy:.2f}")
print("\nMultiple Regression:")
model2 = sm.OLS.from_formula('Age ~ Polyuria + weakness + Polydipsia', df1).fit()
print(model2.summary())
#---> Comparing all
print("\nPima Dataset:")
print(model1.params)
print("\nUCI Dataset:")
print(model2.params)
