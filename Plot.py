import  matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np

from Config import  Config

#data = pd.read_csv("./pollution.csv")
# columns = data.columns
# print(columns)
# for i in range(1,len(columns)):
#     plt.plot(data.iloc[:,i])
#     plt.title(columns[i])
#     plt.show()

config = Config()


hat_y = np.load(config.path + config.dimname + "-hat_y.npy")
test_y= np.load(config.path + config.dimname + "-test_y.npy")
plt.plot(test_y, label ="test_y")
#plt.plot(hat_y, label ="hat_y")
plt.legend()
plt.show()

