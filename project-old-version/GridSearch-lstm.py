
import os
import  pandas as pd
import  numpy as np
from lstm.LSTM_Interface import  startTrainMult
from Config import Config

import  pandas as pd
import  numpy as np
from  sklearn import  metrics
from  lstm.Predict_Interface import  PredictWithModel

import  csv

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
    #删除y_test 为0元素
    zero_index = np.where(y_test == 0)
    y_hat = np.delete(y_hat,zero_index[0])
    y_test = np.delete(y_test,zero_index[0])
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

config = Config()
layer_grid = [32,48,64,80,96,112]

path = config.multpath
print(path)
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)


data = pd.read_csv("./pollution.csv")
#注:为了演示方便故不使用wnd_dir，其实可以通过代码将其转换为数字序列
data = data.drop(['wnd_dir'], axis = 1)
#训练数据
train_data = data.iloc[:int(0.8*data.shape[0]),:]
print("训练数据长度为",train_data.shape[0])

#选取后20%
test_data = data.iloc[int(0.8*data.shape[0]):,:]
print("测试数据长度为",test_data.shape[0])

name = config.dimname

##csv文件 保存网格搜索信息
csvfile = open('./gridsearch.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(csvfile)
csv_writer.writerow(["layer0","layer1","RMSE","MAE","MAPE"])
csvfile.close()

for i in range(len(layer_grid)):
    for j in range(len(layer_grid)):
        config.change_lstm_layers([layer_grid[i],layer_grid[j]])
        print(config.lstm_layers)

        ## 进行训练
        model, normalize = startTrainMult(train_data, name, config)
        model.save(config.multpath + "model" +"-"+str(layer_grid[i])+"-"+str(layer_grid[j]) + ".h5")
        #归一化文件都是一样的 严格来讲 命名有优化空间
        np.save(config.multpath + name + ".npy", normalize)



        ##进行测试
        y_hat, y_test = PredictWithModel(test_data, name, model, normalize, config)
        y_hat = np.array(y_hat, dtype='float64')
        y_test = np.array(y_test, dtype='float64')

        print(y_hat.shape, y_test.shape)

        rmse = GetRMSE(y_hat, y_test)
        mae = GetMAE(y_hat, y_test)
        mape = GetMAPE_Order(y_hat, y_test)

        print("评价指标为",rmse,mae,mape)

        #csv_writer.writerow(["layer0", "layer1", "RMSE", "MAE", "MAPE"])
        #只是为了每训练一次实时保存
        csvfile = open('./gridsearch.csv','a+',encoding='utf-8',newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(layer_grid[i]),str(layer_grid[j]),str(rmse),str(mae),str(mape)])
        csvfile.close()

        np.save(config.multpath + "y_hat" + "-"+str(layer_grid[i])+"-"+str(layer_grid[j]) + ".npy", y_hat)
        np.save(config.multpath + "y_test" + "-"+str(layer_grid[i])+"-"+str(layer_grid[j]) + ".npy", y_test)
        print("结束")



