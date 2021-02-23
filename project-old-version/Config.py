#使用类实现一个配置文件
class Config:
    def __init__(self):
        self.multpath = './Model/'
        self.dimname = 'pollution'
        #使用前n_predictions 步去预测下一步
        self.n_predictions = 30

        #指定LSTM两层的神经元个数
        self.layers = [80,80]
        self.dropout = 0.2

        self.batch_size = 64
        self.optimizer = 'adam'
        self.loss_metric = 'mse'
        self.validation_split = 0.2
        self.verbose = 1
        self.epochs = 200


    ## 是一个数组 [64,64]
    def change_lstm_layers(self,layers):
        self.layers = layers

