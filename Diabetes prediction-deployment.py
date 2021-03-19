import numpy as np
import pandas as pd

dataset = pd.read_csv('kaggle_diabetes.csv')

dataset = dataset.rename(columns={'DiabetesPedigreeFunction':'DPF'})

dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']] = dataset[['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI']].replace(to_replace=0, value=np.NaN )

dataset['Glucose'].fillna(dataset['Glucose'].mean(), inplace=True)
dataset['BloodPressure'].fillna(dataset['BloodPressure'].mean(), inplace=True)
dataset['SkinThickness'].fillna(dataset['SkinThickness'].median(), inplace=True)
dataset['Insulin'].fillna(dataset['Insulin'].median(), inplace=True)
dataset['BMI'].fillna(dataset['BMI'].median(), inplace=True)

from sklearn.model_selection import train_test_split
X = dataset.drop(columns='Outcome')
Y = dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, Y_train)

import pickle
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier,open(filename,'wb'))

                                                                             