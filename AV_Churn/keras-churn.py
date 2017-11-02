import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml

from sklearn.model_selection import train_test_split
from sklearn import metrics 

import numpy
import pandas as pd

df_train = pd.read_csv('num-train.csv')
df_test = pd.read_csv('num-test.csv')

test_actual = pd.read_csv('test.csv')
uid = test_actual['UCIC_ID']

# # split into input (X) and output (y) variables
# X = df_train.drop(['Responders'], axis = 1)
# y = df_train.Responders.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.9)

# # create model
# model = Sequential()
# model.add(Dense(400, input_dim=204, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X_train.values, y_train, epochs=10, batch_size=10)
# # evaluate the model
# scores = model.evaluate(X_test.values, y_test)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# model.fit(X.values, y, epochs=10, batch_size=10)

# try : 
#     y_pred = model.predict(df_test.values)
#     print(type(y_pred))
#     subm = pd.DataFrame({'UCIC_ID':uid.values, 'Responders':y_pred})
#     subm.set_index('UCIC_ID', inplace = True)
#     subm.to_csv('mlp-keras-submission.csv')
# except Exception as e:
#     print(e)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

predictions = loaded_model.predict(df_test.values)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open("model.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)

