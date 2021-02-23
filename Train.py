import tensorflow as tf
import numpy as np
import os
from  Config import  Config
from  lstm.LSTM_Interface import  start_Train
import pandas as pd

config = Config()

if not os.path.exists(config.path): os.makedirs(config.path)


data = pd.read_csv("./pollution.csv")
#注:为了演示方便故不使用wnd_dir，其实可以通过代码将其转换为数字序列
data = data.drop(['wnd_dir'], axis = 1)
data = data.iloc[:int(0.8*data.shape[0]),:]



model,normalize = start_Train(data,config)

model.save(config.path+config.dimname+".h5")
np.save(config.path+config.dimname+".npy",normalize)
