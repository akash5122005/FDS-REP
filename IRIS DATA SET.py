import pandas as pd
import matplotlib.pyplot as pl
df=pd.read_csv("Iris_Data1.csv")
print(df)
print()
print("Descriptive analysis on Iris Data: ")
print("Info:\n")
print(df.info)
print("Describe:\n")
print(df.describe)
print("Head:\n")
print(df.head)
print("Species count:\n")
print(df['CLASS'].value_counts())
print("IsNull:\n")
print(df.isnull)
print("Max:\n")
print(df.max)
print("Shape:\n")
print(df.shape)
print("Size:\n")
print(df.size)
#plotting graph
x = df['ID'].head(10)
y = df["SL"].head(10)
print(pl.title("Bar Graph - ID vs Sepal Length"))
print(pl.plot(x,y,marker = "*",linestyle = "dashed"))
pl.xlabel("ID")
pl.ylabel("Sepal length")
pl.show()