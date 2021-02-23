import numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from lstm.LSTM_Interface import create_dataset

# 设定为自增长
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth = True
session = tf.Session(config=configtf)
KTF.set_session(session)


# 指定传入该单维的最大最小值
def FNormalize_Single(data, norm):
    listlow = norm[0]
    listhigh = norm[1]
    delta = listhigh - listlow
    if delta != 0:
        for i in range(len(data)):
            data[i, 0] = data[i, 0] * delta + listlow
    return data


# 多步预测的归一化 但是同样适用于单步
def FNormalize_MultSteps(data, norm):
    listlow = norm[0]
    listhigh = norm[1]
    delta = listhigh - listlow
    if delta != 0:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = data[i, j] * delta + listlow
    return data


# 使用训练数据的归一化
def NormalizeMultUseData(data, normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        # 第i列
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            # 第j行
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return data


def PredictWithModel(data, name, model, normalize, config):
    data = data.iloc[:, 1:]
    print(data.columns)

    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # 归一化
    data = NormalizeMultUseData(data, normalize)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _= create_dataset(data, config.n_predictions)
    _, testY = create_dataset(data_y, config.n_predictions)

    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)


    print("testX Y shape is:", testX.shape, testY.shape)

    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)

    # 加载模型
    y_hat = model.predict(testX)
    print('预测成功')

    # 反归一化
    testY = FNormalize_MultSteps(testY, normalize[yindex,])
    y_hat = FNormalize_MultSteps(y_hat, normalize[yindex,])
    return y_hat, testY
