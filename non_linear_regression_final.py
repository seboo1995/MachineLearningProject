#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:14:06 2019

@author: sebair
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv("/home/sebair/Desktop/dataset/Admission_Predict_Ver1.1.csv")

X = data.iloc[:,0:8]
print(X.shape)
y = data.iloc[:,8:9]
X = np.array(X)
y = np.array(y)


#print("New Array: ",new_array)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
new_array= np.zeros((len(X_train),36))
new_array[:,0:8] = X_train[:,0:8]

print("Before : This is new :",new_array[0])
k = 8
#print(len(data.columns))
for x in range(1,len(data.columns)-1):
    for y in range(1,len(data.columns)-1):
        if(x > y or x == y):
            new_array[:,k] = new_array[:,x] * new_array[:,y]
            
            k = k + 1



new_array_test= np.zeros((len(X_test),36))
new_array_test[:,0:8] = X_test[:,0:8]

m = 8
for x in range(1,len(data.columns)-1):
    for y in range(1,len(data.columns)-1):
        if(x > y or x == y):
            new_array_test[:,m] = new_array_test[:,x] * new_array_test[:,y]
            
            m = m + 1
        
dim = new_array_test.shape[1]





new_array[:,:1] = np.ones((len(new_array),1))# adding the bias
new_array_test[:,:1] = np.ones((len(new_array_test),1))#adding the bias

grammian_matrix = np.zeros((dim,dim))#(36 36) len(X_train[:,1])
for x in range(36):
    for y in range(x+1):
        temp = np.dot(new_array[:,x],new_array[:,y])
        grammian_matrix[x][y] = temp
        grammian_matrix[y][x] = temp

res = np.zeros((36,1))#change 36,1

for i in range(36):
    temp = np.dot(y_train[:,0],new_array[:,i])
    res[i][0]= temp


    
#print(grammian_matrix.shape)
#print(res.shape)
output = np.linalg.solve(grammian_matrix,res)
#print(output)
hits = 0
error = 0
f = open("results_from_NLR_admission.txt","w+")
Y_pred = np.matmul(new_array_test,output)
for x in range(len(Y_pred)):
    f.write(("Predicted:{:.2f} ".format((Y_pred[x][0])) +"Real:{:.1f} ".format((y_test[x][0])) +"Absolute Error: "+str(Y_pred[x][0] - y_test[x][0]) +  "\n"))
    temp = (Y_pred[x][0] - y_test[x][0])**2
    error = error + temp
    if (abs(Y_pred[x][0] - y_test[x][0]) < 0.03):
            hits+=1
f.write(("The final cost is NLR : "+str(hits/len(y_test))))
f.close()




































