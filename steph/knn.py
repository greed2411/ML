from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('data.csv')
#print(df.head())
# create design matrix X and target vector y
X = np.array(df.ix[:, 0:13]) 	# end index is exclusive
y = np.array(df['income']) 	# another way of indexing a pandas df
#print(X)
#print(y)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=30, weights = 'distance')

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print (accuracy_score(y_test,pred))
#print(pred)
