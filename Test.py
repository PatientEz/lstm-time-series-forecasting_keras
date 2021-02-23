import  pandas as pd
import  numpy as np
from  sklearn import  metrics
from Config import  Config
from  lstm.Predict_Interface import  Predict
from keras.models import  load_model



def GetRMSE(y_hat,y_test):
    sum = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
    return  sum

def GetMAE(y_hat,y_test):
    sum = metrics.mean_absolute_error(y_test, y_hat)
    return  sum

def GetMAPE(y_hat,y_test):
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

def GetMAPE_Order(y_hat,y_test):
    #删除test_y 为0元素
    zero_index = np.where(y_test == 0)
    y_hat = np.delete(y_hat,zero_index[0])
    y_test = np.delete(y_test,zero_index[0])
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum


config = Config()

data = pd.read_csv("./pollution.csv")
data = data.drop(['wnd_dir'], axis = 1)
#选取后20%
data = data.iloc[int(0.8*data.shape[0]):,:]
print("长度为",data.shape[0])

normalize = np.load(config.path+config.dimname+".npy")
model = load_model(config.path+config.dimname+".h5")

hat_y, test_y = Predict(data, model, normalize, config)
hat_y = np.array(hat_y, dtype='float64')
test_y = np.array(test_y, dtype='float64')

print("RMSE为", GetRMSE(hat_y, test_y))
print("MAE为", GetMAE(hat_y, test_y))
#print("MAPE为",GetMAPE(hat_y,test_y))
print("MAPE为", GetMAPE_Order(hat_y, test_y))

np.save(config.path + config.dimname +"-hat_y.npy", hat_y)
np.save(config.path + config.dimname +"-test_y.npy", test_y)
print("结束")
