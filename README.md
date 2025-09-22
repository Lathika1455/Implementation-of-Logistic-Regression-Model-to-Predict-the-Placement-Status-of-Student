# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values

4.Using logistic regression find the predicted values of accuracy , confusion matrices

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LATHIKA SREE R
RegisterNumber: 212224040169 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Show first 5 rows of raw data
print("data.head")
print(data.head(), "\n")

# Copy dataset and drop sl_no & salary
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print("data1.head")
print(data1.head(), "\n")

# Check for missing values
print("isnull()")
print(data1.isnull().sum(), "\n")

# Check for duplicate rows
print("data duplicate ")
print(data1.duplicated().sum(), "\n")

# Label Encoding categorical features
le = LabelEncoder()
for col in ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]:
    data1[col] = le.fit_transform(data1[col])

print("data")
print(data1.head(), "\n")

# Features and Target
X = data1.iloc[:, :-1]
y = data1["status"]

print("status")
print(y.head(), "\n")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Logistic Regression
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

print("y_pred")
print(y_pred, "\n")

# Accuracy
print(" Accuracy Score")
print(accuracy_score(y_test, y_pred), "\n")

# Confusion Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred), "\n")

# Classification Report
print("Classification")
print(classification_report(y_test, y_pred), "\n")

# Custom Prediction Example
sample = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
print("LR predict")
print(lr.predict(sample))
```

## Output:

## data 

<img width="723" height="328" alt="image" src="https://github.com/user-attachments/assets/fa5f1327-3ad1-445f-9578-470e2bd9d9d8" />

## data1.head

<img width="757" height="312" alt="image" src="https://github.com/user-attachments/assets/af3742f5-8689-403d-bee6-a671bc7eb992" />

## isnull()

<img width="363" height="346" alt="image" src="https://github.com/user-attachments/assets/80afb1e7-07c3-4c73-89ce-4d9ac581f1d1" />

## data duplicate

<img width="730" height="64" alt="image" src="https://github.com/user-attachments/assets/bf03435a-2b25-4576-b1a2-428acf9daee5" />

## data

<img width="776" height="338" alt="image" src="https://github.com/user-attachments/assets/22085a72-6f83-40de-a53b-5f26a1c90e12" />

## status

<img width="572" height="175" alt="image" src="https://github.com/user-attachments/assets/e1eadbf8-746b-4cff-8546-1170a270307f" />

## y_pred

<img width="830" height="91" alt="image" src="https://github.com/user-attachments/assets/7d99798d-8eb7-4c7e-b7f8-9df7578513fc" />

## Accuracy value

<img width="673" height="67" alt="image" src="https://github.com/user-attachments/assets/1a606aa1-10e3-4e3c-a0e8-ecda3462a160" />

## Confusion matrix

<img width="560" height="109" alt="image" src="https://github.com/user-attachments/assets/fa91eec0-0c58-4169-a1bb-721735a0f886" />

## classification matrix

<img width="835" height="251" alt="image" src="https://github.com/user-attachments/assets/34bb9433-0065-4367-ab5c-b0367cca4d3b" />

## LR predict

<img width="473" height="58" alt="image" src="https://github.com/user-attachments/assets/c9a705fa-0d5e-4bec-ac9b-53ae722dd07b" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
