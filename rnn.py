


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
#all scaled values will be between 0 and 1

# Creating a data structure with 60 timesteps and 1 output
# Look at the 60 timesteps in the past to get a trend, learn by trial and error
X_train = []
y_train = []
# Start from 60th day and go to 1257 which must be written as 1258
# Memorise the first 60 values to predict the 61st price
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    # x train - in first iteration it takes 60 data points
    y_train.append(training_set_scaled[i, 0])
    # y train - in this the first 59 is used to predict the 60th
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# need to add a new dimesion - for this we must use numpy
# shape() is used to give number of row and column
# RNN accepts only 3d tensor with shape - batch size, timesteps, input_dim
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Building RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Making the predictions and visualising the results

# Getting the real value
dataset_test = pd.read_csv('Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted value
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
