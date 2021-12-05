import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X = pd.read_csv("./Training Data/Linear_X_Train.csv").values
Y = pd.read_csv("./Training Data/Linear_Y_Train.csv").values

theta = np.load("ThetaList.npy")
# 100, 2
T0  = theta[:,0]
T1 = theta[:,1]

plt.ion() # On the interactive mode of matplotlib
# As per our example the loss was decreasing upto 50 iterations
for i in range(0,50,3):
    y_ = T1[i]*X + T0
    #points
    plt.scatter(X,Y)
    # line 
    plt.plot(X,y_,'red')
    plt.draw()
    plt.pause(1) # Pause the graph for 1 sec
    plt.clf() # Destroy the last object
    

