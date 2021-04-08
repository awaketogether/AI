import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set_theme()

#Data computeed and processed.
dataset = pd.read_csv("/SleepAnalyzer/outputAwakeInformation/asleepAwakeInformation.csv")
dataset.drop(['Number', 'timestamp', 'count'], axis=1, inplace=True) #dataset.drop(['Number', 'timestamp'], axis=1, inplace=True)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#plt.scatter(X_train[:, 0], X_train[:, 1], marker='o')

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=25)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classifier.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix

mat = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

print(mat)
print("total number of raw:", len(X_test))
names = np.unique(y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Pre#dicted')
plt.show()

#error = []

# Calculating error for K values between 1 and 40
#for i in range(1, 40):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error.append(np.mean(pred_i != y_test))
#   
#plt.figure(figsize=(12, 6))
#plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
#         markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')
#plt.xlabel('K Value')
#plt.ylabel('Mean Error')

#plt.show()