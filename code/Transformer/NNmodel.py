import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import backend as K
from keras.layers import Dropout


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def trainNN(X_train,X_test,y_train,y_test,epochs,dates,temp_toplot):


    feats = len(X_train[0])
    print(feats)
    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(feats , activation='tanh', input_dim=feats))
    model.add(Dropout(0.1))
    model.add(Dense(units=int(feats), activation='tanh'))
    model.add(Dropout(0.1))
    # model.add(Dense(units=int(feats/2), activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(units=int(feats / 4), activation='relu'))
    # model.add(Dropout(0.1))
    # Adding the output layer
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss="mean_squared_error")

    model.fit(X_train, y_train, batch_size=128, epochs=epochs)

    y_pred = model.predict(X_test)

    plt.plot(dates[-len(y_test):],y_test, color='red', label='Real data')
    plt.plot(dates[-len(y_test):],y_pred, color='blue', label='Predicted data')
    #plt.plot(dates[-len(y_test):],temp_toplot[-len(y_test):],color='green')
    plt.title('Prediction')
    plt.legend()
    plt.show()
    print(forecast_accuracy(y_pred,y_test))


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE

    return ({'mape': mape, 'me': me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse})

def convert_data(data,lags,step):
    Xtrain=[]
    Ytrain=[]
    for i, g in enumerate(data):
        if i >= lags+step:
            last_lags_load = data[i - lags-step:i-step,0]
            temp_week_Month_day_Hour = data[i,2:]
            last_lags_load=np.append(last_lags_load,temp_week_Month_day_Hour)
            Xtrain.append(last_lags_load)
            Ytrain.append(data[i,0])
    return np.array(Xtrain),np.array(Ytrain)






if __name__ == '__main__':

    window=1*24
    step=24
    epochs=20


    dftrain =pd.read_csv('train.csv',index_col=0).fillna(method='bfill')
    dftest =pd.read_csv('test.csv',index_col=0).fillna(method='bfill')
    dftest.index = pd.to_datetime(dftest.index)

    #sc = StandardScaler()
    sc = MinMaxScaler()

    X = sc.fit_transform(dftrain)
    X_test = sc.transform(dftest)



    Xtrain,Ytrain=convert_data(X,window,step)
    Xtest,Ytest=convert_data(X_test,window,step)
    print(len(Xtrain))

    trainNN(Xtrain,Xtest,Ytrain,Ytest,epochs,dftest.index,X_test[:,1])

