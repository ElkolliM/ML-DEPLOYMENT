import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("global_african_crises.csv")

print(df.head())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col])

# Select independent and dependent variable
x=df.drop('banking_crisis',axis=1)
y=df['banking_crisis']

X_means = x.mean()
X_stds = x.std()
X = (x-X_means)/X_stds

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf_pred_model_d1 = RandomForestClassifier(n_estimators=10, min_samples_leaf=20).fit(X_train, y_train)

# Split the dataset into train and test
#evaluate
print("Accuracy:")
print( metrics.accuracy_score(y_test, rf_pred_model_d1.predict(X_test)))
print(confusion_matrix(y_test, rf_pred_model_d1.predict(X_test)))

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, rf_pred_model_d1.predict(X_test)))



# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)




# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))





complete the program should read lines of text from standard input. Each line begins with a positive integer N, the size of the array, followed by a semicolon, followed by a comma separated list of positive numbers ranging from 0 to N-2, inclusive.


 import sys
# import numpy as np
# import pandas as pd
# from sklearn import ...

for line in sys.stdin: print(line, end="")



print(line, end="")