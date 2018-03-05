import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')
adjusted_dataset = dataset.iloc[:, [1, 2,4,5,9]].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Changing label 'male' to 1 and 'female' to 0
labelencoder_X_1 = LabelEncoder()
adjusted_dataset[:, 2] = labelencoder_X_1.fit_transform(adjusted_dataset[:, 2])

#Changing type of data to float64
adjusted_dataset= adjusted_dataset.astype(float)
adjusted_dataset = pd.DataFrame(adjusted_dataset)

#Dropping missing data
adjusted_dataset = adjusted_dataset.dropna()

#Separating dependent and independent values
X=adjusted_dataset.iloc[:, 1:]
y=adjusted_dataset.iloc[:, 0].values

#Creating dummie variablesfro passanger class, ignoring one columnt in order to avoid overfitting
dumm = pd.get_dummies(X.iloc[:,0], drop_first = True)
X = X.drop(X.columns[0], axis = 1)
X = pd.concat([dumm, X], axis = 1)
X = X.values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling in order to avoid euclidean error
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Importing neccesary libraries for neural networks
import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#Adding layers to network
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 5))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Choosing optimizer and loss function for learning 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ['accuracy'])

#Fit model to data
classifier.fit(X_train, y_train, batch_size = 32, epochs = 400)

#Predicting new values from test set
y_pred = classifier.predict(X_test)

#changing predicted values from float to boolean
y_pred = (y_pred > 0.5)

#Veryfing results in confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)