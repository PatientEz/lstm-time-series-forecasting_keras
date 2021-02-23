from  tensorflow.keras import Sequential
from  tensorflow.keras.layers import LSTM,Dense,Activation,Dropout
from  tensorflow.keras.callbacks import History,Callback,EarlyStopping
import  numpy as np



def create_dataset(dataset, n_predictions):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    #这里的 -1 并不必要 但是可以避免某些状况下的bug
    for i in range(len(dataset) -n_predictions - 1):
        dataX.append(dataset[i:(i+n_predictions),:])
        dataY.append(dataset[i+n_predictions,:])
    train_x = np.array(dataX)
    train_y = np.array(dataY)
    return train_x, train_y

#多维归一化
def Normalize_Mult(data):
    normalize = np.zeros((data.shape[1],2),dtype='float64')
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,:] = [listlow,listhigh]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    return  data,normalize


def lstm_model(train_x,train_y,config):

    model = Sequential()
    model.add(LSTM(config.lstm_layers[0],input_shape=(train_x.shape[1],train_x.shape[2]),
                   return_sequences=True))
    model.add(Dropout(config.dropout))

    model.add(LSTM(
        config.lstm_layers[1],
        return_sequences=False))
    model.add(Dropout(config.dropout))

    model.add(Dense(
        train_y.shape[1]))
    model.add(Activation("relu"))

    model.summary()

    cbs = [History(), EarlyStopping(monitor='val_loss',
                                    patience=config.patience,
                                    min_delta=config.min_delta,
                                    verbose=0)]
    model.compile(loss=config.loss_metric,optimizer=config.optimizer)
    model.fit(train_x,
                   train_y,
                   batch_size=config.lstm_batch_size,
                   epochs=config.epochs,
                   validation_split=config.validation_split,
                   callbacks=cbs,
                   verbose=True)
    return model


def start_Train(data,config):
    #删去时间步
    data = data.iloc[:, 1:]
    print(data.columns)

    yindex = data.columns.get_loc(config.dimname)
    data = np.array(data, dtype='float64')

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    # 数据归一化
    data, normalize = Normalize_Mult(data)
    data_y = data[:, yindex]
    if len(data_y.shape) == 1:
        data_y = data_y.reshape(data_y.shape[0], 1)

    # 构造训练数据
    train_x, _ = create_dataset(data, config.n_predictions)
    _, train_y = create_dataset(data_y, config.n_predictions)
    print("train_x y shape is:", train_x.shape, train_y.shape)

    # 进行训练
    model = lstm_model(train_x,train_y,config)

    return  model,normalize