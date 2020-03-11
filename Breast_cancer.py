# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:55:05 2020

@author: User
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,2:32].values
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Exploring the Data

#Radius Mean
plt.hist(X_train[:, 0], bins = 25, width = 0.7, color = 'teal')
plt.title('Radius Mean')
plt.xlabel('Size')
plt.ylabel('Frequency')

#Texture Mean
plt.hist(X_train[:, 1], bins = 25, width = 0.7, color = 'teal')
plt.title('Texture Mean')
plt.xlabel('Standard Deviation of Grey Scale Values ')
plt.ylabel('Frequency')

#Perimeter Mean
plt.hist(X_train[:, 2], bins = 25, width = 4.5, color = 'teal')
plt.title('Perimeter Mean')
plt.xlabel('Mean Size of the Core Tumor')
plt.ylabel('Frequency')

#Area Mean
plt.hist(X_train[:, 3], bins = 25, width = 70, color = 'teal')
plt.title('Area Mean')
plt.xlabel('Mean Area of the Core Tumor')
plt.ylabel('Frequency')

#Smoothness Mean
plt.hist(X_train[:, 4], bins = 25, width = 0.003, color = 'teal')
plt.title('Smoothness Mean')
plt.xlabel('Mean of Local Variation in Radius Lengths')
plt.ylabel('Frequency')

#Compactness Mean
plt.hist(X_train[:, 5], bins = 25, width = 0.008, color = 'teal')
plt.title('Compactness Mean')
plt.xlabel('Mean of Perimeter^2 / area - 1.0')
plt.ylabel('Frequency')

#Concavity Mean
plt.hist(X_train[:, 6], bins = 25, width = 0.012, color = 'teal')
plt.title('Compactness Mean')
plt.xlabel('Mean of Severity of Concave Portions of the Contour')
plt.ylabel('Frequency')

#Concave Points
plt.hist(X_train[:, 7], bins = 25, width = 0.0065, color = 'teal')
plt.title('Concave Points')
plt.xlabel('Mean for Number of Concave Portions of the Contour')
plt.ylabel('Frequency')

#Symmetry Mean
plt.hist(X_train[:, 8], bins = 25, width = 0.0062, color = 'teal')
plt.title('Symmetry Mean')
plt.xlabel('Symmetry Mean')
plt.ylabel('Frequency')

#Fractal Dimension Mean
plt.hist(X_train[:, 9], bins = 25, width = 0.0014, color = 'teal')
plt.title('Fractal Dimension Mean')
plt.xlabel('Mean for "Coastline Approximation" - 1')
plt.ylabel('Frequency')

#Create Heatmap to See Correlation Between Independent Variables

plt.subplots(figsize=(8, 5))
sns.heatmap(X_train.corr(), annot=True, cmap="RdYlGn")
plt.show()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()