import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loc = r'E:\Mihir_Data\aa_06_ML_AI\Python_Tests\Flask'
df=pd.read_csv(loc + '/Iris_Data.csv')

df.columns

X= df[['SepalLength', 'sepalWidth', 'PetalLength', 'PetalWidth']]
y= df['Species']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# Train the model on our train dataset
lr.fit(X,y)

# Train the model with the training set

lr.fit(X_train,y_train)

y_test_hat=lr.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat)*100,'%')
