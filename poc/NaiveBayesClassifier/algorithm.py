import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import sys

if not (sys.version_info.major == 3 and sys.version_info.minor >= 5):
    from sklearn.externals import joblib 
    
#Data computeed and processed.
df = pd.read_csv('data.csv')
df.drop(['Number', 'timestamp'], axis=1, inplace=True)

input = df.drop('sleep', axis=1)
target = df.sleep

X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.25, random_state=0)

model = GaussianNB() 
model.fit(X_train, y_train)

##### Print model viability/score

#print(model.score(X_test, y_test))
#print(y_test[:10])
#print(model.predict(X_test[:10]))

##### Save the model as a pickle in a file
#if not (sys.version_info.major == 3 and sys.version_info.minor >= 5):
    #joblib.dump(model, 'modelSaved.pkl') 
  
###### Preditect output & Plot Matrix confusion
pred = model.predict(X_test)

# Plot & Confusion Matrix
mat = confusion_matrix(pred, y_test)
print(mat)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()
print("total number of raw:", len(X_test))