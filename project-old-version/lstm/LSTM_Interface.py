import  os
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.layers import CuDNNLSTM
from keras.layers import LSTM

from keras.callbacks import CSVLogger

import keras.backend.tensorflow_backend as KTF



#设定为自增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


def create_dataset(dataset, n_predictions):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    #这里的 -1 并不必要 但是可以避免某些状况下的bug
    for i in range(len(dataset) - n_predictions - 1):
        dataX.append(dataset[i:(i + n_predictions), :])
        dataY.append(dataset[i + n_predictions, :])
    TrainX = np.array(dataX)
    TrainY = np.array(dataY)

    return TrainX, TrainY


def trainModel(trainX,trainY,config):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    path: 存储模型文件路径
    name: 存储模型文件名称
    config:  配置文件
    '''
    # CPU版本需要将CuDNNLSTM 改为LSTM
    model = Sequential()
    model.add(CuDNNLSTM(
        config.layers[0],
        input_shape=(trainX.shape[1], trainX.shape[2]),
        return_sequences=True))
    model.add(Dropout(config.dropout))

    model.add(CuDNNLSTM(
        config.layers[1],
        return_sequences=False))
    model.add(Dropout(config.dropout))

    model.add(Dense(
        trainY.shape[1]))
    model.add(Activation("relu"))

    model.summary()
    model.compile(loss=config.loss_metric, optimizer=config.optimizer)
    #csvlogger = CSVLogger("./lstmlog.csv", separator=',', append=False)
    #model.fit(trainX, trainY, batch_size=config.batch_size, epochs=config.epochs,validation_split=config.validation_split, verbose=config.verbose,callbacks=[csvlogger])
    model.fit(trainX, trainY, batch_size=config.batch_size, epochs=config.epochs,validation_split=config.validation_split, verbose=config.verbose)

    return model



#多维归一化
def NormalizeMult(data):
    # data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize



def startTrainMult(data,name,config):
    '''
    data: 多维数据
    返回训练好的模型
    '''
    data = data.iloc[:,1:]
    print(data.columns)

    yindex = data.columns.get_loc(name)
    data = np.array(data,dtype='float64')

    if len(data.shape) == 1:
        data = data.reshape(-1,1)
    #数据归一化
    data, normalize = NormalizeMult(data)
    data_y = data[:,yindex]
    data_y = data_y.reshape(data_y.shape[0],1)
    print(data.shape, data_y.shape)

    #构造训练数据
    trainX, _ = create_dataset(data, config.n_predictions)
    _,trainY = create_dataset(data_y,config.n_predictions)
    print("trainX Y shape is:",trainX.shape, trainY.shape)


    if len(trainY.shape) == 1:
        trainY = trainY.reshape(-1,1)

    print("trainX Y shape is:",trainX.shape,trainY.shape)

    # 进行训练
    model = trainModel(trainX, trainY, config)

    return model,normalize


