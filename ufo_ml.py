"""
The goal of this ML is to predict the season of the sighting based on the time and location.
"""

import pandas as pd

df = pd.read_csv("ufo_data.csv")

# create seasons column
seasons = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer',
           7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall', 12: 'winter'}

# create a new column 'season' by mapping month numbers to the corresponding seasons
df['Season'] = df['Month'].map(seasons)

# Applying one hot encoding on cateogrical columns...
# CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder()

le = LabelEncoder()
df['City'] = le.fit_transform(df['City'])
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(df[['City']])

df['Country'] = le.fit_transform(df['Country'])
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(df[['Country']])

df['Shape of UFO'] = le.fit_transform(df['Shape of UFO'])
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(df[['Shape of UFO']])

df['State'] = le.fit_transform(df['State'])
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(df[['State']])

df['Season'] = le.fit_transform(df['Season'])
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(df[['Season']])

x = df.iloc[:, [2,3,4,5,6,7,8]].values
y = df.iloc[:, -1].values

# split dataset
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

# CLASSIFIER 1: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
forest.fit(x_train, y_train)

y_pred_forest = forest.predict(x_test)

# CLASSIFIER 2: NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB()
naiveBayes.fit(x_train, y_train)

y_pred_bayes = naiveBayes.predict(x_test)

# CLASSIFIER 3: KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 6, metric = "minkowski", p = 2)
KNN.fit(x_train, y_train)

y_pred_knn = KNN.predict(x_test)

# CLASSIFER 4: Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

# PREDICTIONS
from sklearn.metrics import confusion_matrix
cm_forest = confusion_matrix(y_test, y_pred_forest)
cm_naive = confusion_matrix(y_test, y_pred_bayes)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_lr = confusion_matrix(y_test, y_pred_lr)

from sklearn.metrics import accuracy_score
acc_forest = accuracy_score(y_test, y_pred_forest)
acc_naive = acc_tree = accuracy_score(y_test, y_pred_bayes)
acc_knn = acc_tree = accuracy_score(y_test, y_pred_knn)
acc_lr = acc_tree = accuracy_score(y_test, y_pred_lr)

print("Accuracy of Random Forest: " + str(acc_forest))
print("Accuracy of Naive Bayes: " + str(acc_naive))
print("Accuracy of KNN: " + str(acc_knn))
print("Accuracy of Logistic Regression: " + str(acc_lr))