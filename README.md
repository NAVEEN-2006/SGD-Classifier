# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Load Data: Load the Iris dataset using load_iris() and create a pandas DataFrame with the feature names and target variable.


2.Prepare Features and Target: Split the DataFrame into features (X) and target (y) by dropping the target column from X.


3.Split Data: Use train_test_split to divide the dataset into training and testing sets with a test size of 20%.


4.Train Model: Initialize and fit a Stochastic Gradient Descent (SGD) classifier on the training data.


5.Evaluate Model: Predict the target values for the test set, calculate accuracy, and print the confusion matrix to assess the model's performance. 


## Program:

```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: NAVEEN KUMAR S
RegisterNumber:  212223040129
```


```
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head()) 
X = df.drop('target',axis=1) 
X
y=df['target']
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test) 
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}") 
cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

## Output:


## Dataset :


![image](https://github.com/user-attachments/assets/3c74c116-1c77-4168-8865-a658323bd4b2)


## X and Y values :


![image](https://github.com/user-attachments/assets/e4ad2ced-d1a3-4e74-9266-eef804b64b2f)



## Model:


![image](https://github.com/user-attachments/assets/51dcc037-564a-4e32-8caf-9635d3698421)


## Accuracy:


![image](https://github.com/user-attachments/assets/77428461-cb2b-42e2-985a-7903b98327b2)


## Matrix:


![image](https://github.com/user-attachments/assets/d821ecd2-2158-4d87-a66a-b81f29f0d490)


## Confusion Matrix:


![image](https://github.com/user-attachments/assets/f697f3ab-009a-4bf6-a772-c557f8075c55)


## Result:


Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
