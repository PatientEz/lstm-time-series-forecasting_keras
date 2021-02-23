import numpy as np
import tensorflow as tf
from lstm.LSTM_Interface import create_dataset



def FNormalize(data, norm):
    listlow,listhigh= norm[0],norm[1]
    delta = listhigh - listlow
    if delta != 0:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = data[i, j] * delta + listlow
    return data

# 使用训练数据的归一化
def Normalize_Data(data, normalize):
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


def Predict(data, model, normalize, config):
    #删去时间列
    data = data.iloc[:, 1:]
    yindex = data.columns.get_loc(config.dimname)

    data = np.array(data, dtype='float64')
    # 使用训练数据边界进行的归一化
    data = Normalize_Data(data, normalize)

    data_y = data[:, yindex]
    if len(data_y.shape) == 1:
        data_y = data_y.reshape(-1, 1)

    test_x, _ = create_dataset(data, config.n_predictions)
    _, test_y = create_dataset(data_y, config.n_predictions)

    print("test_x y shape is:", test_x.shape, test_y.shape)
    # 加载模型
    hat_y = model.predict(test_x)
    print('预测成功')

    # 反归一化
    # 命名规则?
    test_y = FNormalize(test_y, normalize[yindex,])
    hat_y = FNormalize(hat_y, normalize[yindex,])
    return test_y,hat_y
