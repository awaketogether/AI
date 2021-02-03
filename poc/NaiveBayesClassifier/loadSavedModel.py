import pandas as pd
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

model = joblib.load('modelSaved.pkl')  
df = pd.read_csv('data_Patient_2.csv')

input = df.drop(['sleep', 'Number', 'timestamp'], axis=1)

# Use the loaded model to make predictions 
pred = model.predict(input) 
print(pred)
mat = confusion_matrix(pred, df.sleep)
print(mat)