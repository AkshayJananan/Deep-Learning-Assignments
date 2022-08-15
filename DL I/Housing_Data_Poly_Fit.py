##AKSHAY J
##21105012
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

##Importing the dataset
data=pd.read_csv('housing.csv')
X=data.iloc[:,0:13].values
Y=data.iloc[:,13].values

##Splitting the data into testing and training
from sklearn.model_selection import train_test_split

X_train,X_test,t_train,t_test=train_test_split(X,Y,test_size=0.1,random_state=0)

#INITIALISING
w = np.ones(14)
y = np.zeros(len(t_train))
y_pred = np.zeros(len(t_test))
eta = 10 **-8
x_T = pd.DataFrame(X_train)
x_T_test = pd.DataFrame(X_test)
one1=np.ones(len(X_train))
one2=np.ones(len(X_test))

#CALCULATION
x_T.insert(0,'',value=one1)
x_T_test.insert(0,'',value=one2)
x_T=x_T.to_numpy(dtype ='float32')
x_T_test=x_T_test.to_numpy(dtype ='float32')
for k in range(1000):
    y = np.matmul(x_T, w)

    der = np.matmul(np.transpose(x_T), np.subtract(y,t_train))

    w = np.subtract(w, eta * der)

print('Weight values:- [w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13]= ',w)

y_pred = np.matmul(x_T_test,w)

#MEAN ABSOLUTE ERROR
MAE=0
for i in range(len(t_test)):

    MAE=MAE+np.abs(t_test[i]-y_pred[i])

print('Mean Absolute Error: ',MAE/len(t_test))

#TESTING THE ACCUARCY USING ONE OF THE TEST VALUE
print('ACTUAL VALUE OF 47th TEST DATA: ',t_test[48])
print("PREDICTED VALUE OF 47th TEST DATA: ",y_pred[48])

#PLOTTING
plt.scatter(X_test[:,6], t_test,color='r',label='Target Value')

plt.scatter(X_test[:,6], y_pred,color='g',label='Predicted Value')

plt.title('Prediction of Housing data using Polynomail fit plotted with 5th input')

plt.xlabel("5th INPUTS IN THE HOUSING DATA---->")

plt.ylabel('TARGET/PREDICTED VALUES---->')

plt.legend(loc='upper right')

plt.show()
