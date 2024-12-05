import matplotlib
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as pl
df=pd.read_csv("diabetes_data_upload.csv")
print(df)
#normal curve plot
data=df['Polyuria']
mean=data.mean()
std_dev=data.std()
random=np.random.normal(mean,std_dev,len(data))
xmin,xmax=pl.xlim()
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x,mean,std_dev)
pl.title("Normal Curve")
pl.plot(x,p,"g",linewidth=2,label="Normal")
pl.show()
#Density and Contour plots
sns.kdeplot(data=df, x='Age', y='weakness', fill=True)
pl.title('Density and Contour Plot')
pl.xlabel('x')
pl.ylabel('y')
pl.show()
#correlation and scatter plot
x = df['Age']
y = df['Obesity']
sns.scatterplot(x)
corr = np.corrcoef(x, y)[0, 1]
pl.title('Scatter Plot with Correlation Coefficient')
pl.xlabel('Age')
pl.ylabel('Obesity')
pl.text(0.5, 0.5, 'Correlation Coefficient: {0:.2f}'.format(corr))
pl.show()
#Histogram plot
numbers = df['Age']
pl.hist(numbers, bins=10)
pl.title("Histogram")
pl.xlabel("Interval")
pl.ylabel("Age")
pl.show()
#3d plot 1
fig = pl.figure()
ax = pl.axes(projection='3d')
zline = np.linspace(0, 5, 20)
xline = df['Age'].head(20)
yline = df['Itching'].head(20)
ax.scatter3D(xline, yline, zline, 'greenmaps')
pl.show()
#3d plot 2
fig = pl.figure()
ax = pl.axes(projection='3d')
zline = np.linspace(0, 5, 20)
xline = df['Age'].head(20)
yline = df['Itching'].head(20)
ax.plot3D(xline, yline, zline)
pl.show()
#3d plot 3
def f(x, y):
return np.sin(np.sqrt(x ** 2 + y ** 2))
x =df['Age'].head(10)
y = df['Obesity'].head(10)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = pl.figure()
ax = pl.axes(projection ='3d')
ax.plot_wireframe(X, Y, Z, color ='green')
ax.set_title('WireFrame for UCI Dataset')
pl.show()
