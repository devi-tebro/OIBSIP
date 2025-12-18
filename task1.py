#task is to train a ml model that can learn from the measurements of the iris species and classify them.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

iris=load_iris()
print(iris.keys())

x=iris.data
y=iris.target

print(x.shape)
print(y.shape)

print(np.unique(y))
print(iris.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler= StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)

model=LogisticRegression(max_iter=200)
model.fit(x_train_scaled,y_train)
pred=model.predict(x_test_scaled)
print("Accuracy:",accuracy_score(y_test,pred))
print("Report:",classification_report(y_test,pred,target_names=iris.target_names))

new_flower=[[5.1, 3.5, 1.4, 0.2]]
new_flower_scaled=scaler.transform(new_flower)
prediction=model.predict(new_flower_scaled)
print("Predicted class:", iris.target_names[prediction[0]])

user_input=input(
    "Enter measurements in order:\n"
    "1) Sepal length  2) Sepal width  3) Petal length  4) Petal width\n"
    "Separate values with space: "
)
new_flower2=[float(x) for x in user_input.split()]
new_flower2=[new_flower2]
new_flower2_scaled=scaler.transform(new_flower2)
prediction2=model.predict(new_flower2_scaled)
print("Predicted species:", iris.target_names[prediction2[0]])
