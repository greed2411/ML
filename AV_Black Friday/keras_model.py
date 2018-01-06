import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

X = pd.read_csv('X.csv')
test_X = pd.read_csv('test_X_dash.csv')
y = X['Purchase']
X.drop('Purchase', axis =1, inplace = True)
X = X.as_matrix()

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)


# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=9, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 7
np.random.seed(seed)

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5,verbose=0)

estimator.fit(X, y)
y_pred = estimator.predict(test_X.as_matrix())

# print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

test = pd.read_csv('test.csv')
test_output = test[['User_ID', 'Product_ID']]
test_output['Purchase'] = y_pred
test_output.to_csv('keras-baseline.csv')