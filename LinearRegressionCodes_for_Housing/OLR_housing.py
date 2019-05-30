#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:14:00 2019

@author: sebair
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
data = pd.read_csv("/home/sebair/Desktop/dataset/house.csv")
X = data.iloc[:,3:len(data.columns)-1]
y = data.iloc[:,len(data.columns)-1:len(data.columns)]
X = np.array(X)
y = np.array(y)
print(len(data.columns)-4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
X_train[:,:1] = np.ones((len(X_train),1))
X_test[:,:1] = np.ones((len(X_test),1))

grammian_matrix = np.zeros((len(data.columns)-4,len(data.columns)-4))#len(X_train[:,1])
for x in range(len(data.columns)-4):
    for y in range(x+1):
        temp = np.dot(X_train[:,x],X_train[:,y])
        grammian_matrix[x][y] = temp
        grammian_matrix[y][x] = temp

res = np.zeros((len(data.columns)-4,1))
print(y_train.shape)
for i in range(len(data.columns)-4):
    temp = np.dot(y_train[:,0],X_train[:,i])
    res[i][0]= temp


    
#print(grammian_matrix.shape)
#print(res.shape)
output = np.linalg.solve(grammian_matrix,res)
#print("Weights:",output)
f= open("results_from_OLR_housing.txt","w+")

error = 0
Y_pred = np.matmul(X_test,output)
for x in range(len(Y_pred)):
    f.write(("Predicted:{:.2f} ".format((Y_pred[x][0])) +"Real:{:.1f} ".format((y_test[x][0])) +"Absolute Error: "+str(Y_pred[x][0] - y_test[x][0]) +  "\n"))
    if (abs(Y_pred[x][0] - y_test[x][0]) < 0.03):
        error+=1

print("The final cost wiht OLR is : ",error/len(y_test))






























