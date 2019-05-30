import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
data = pd.read_csv("/home/sebair/Desktop/dataset/Admission_Predict_Ver1.1.csv")

X = data.iloc[:,2:len(data.columns)-1]
y = data.iloc[:,len(data.columns)-1:len(data.columns)]
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train,y_train)
Y_pred = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,Y_pred))

  
#plot.scatter(X_train, y_train, color = 'red')
#plot.plot(X_train, regr.predict(X_train), color = 'blue')
#plot.show()
print("Score",regr.score(X_test,y_test))




f = open("results_from_sckit_admission.txt","w+")

hits = 0
error = 0

for x in range(len(Y_pred)):
    f.write(("Predicted:{:.2f} ".format((Y_pred[x][0])) +"Real:{:.1f} ".format((y_test[x][0])) +"Absolute Error: "+str(Y_pred[x][0] - y_test[x][0]) +  "\n"))
    temp = (Y_pred[x][0] - y_test[x][0])**2
    error = error + temp
    if (abs(Y_pred[x][0] - y_test[x][0]) < 0.03):
            hits+=1
f.write(("The final cost is sckit_learn : "+str(hits/len(y_test))))
f.close()
