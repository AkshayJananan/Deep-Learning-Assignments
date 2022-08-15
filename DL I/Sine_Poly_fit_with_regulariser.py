##AKSHAY J
##21105012
import numpy as np
from matplotlib import pyplot as plt

##Importing the dataset

data=np.genfromtxt ("Sinedata.dat")
X=data[0,:]
T=data[1,:]

##Splitting the data into testing and training
from sklearn.model_selection import train_test_split

X_train,X_test,t_train,t_test=train_test_split(X,T,test_size=0.1,random_state=0)

#INTIALIZING
w = np.ones(5)
y = np.zeros(len(t_train))
y_pred = np.zeros(len(t_test))
eta = 10 **-3
x_T = np.zeros([len(t_train), len(w)])
x_T_test = np.zeros([len(t_test), len(w)])
lam=1.8

#CALCULATON
for j in range(len(w)):
    for i in range(len(t_train)):

        x_T[i,j]=X_train[i]**j

for k in range(1000):

    y = np.matmul(x_T, w)

    der=(np.matmul(np.transpose(x_T),np.subtract(y,t_train)))+lam*w

    w = np.subtract(w,eta*der)

print('Weight values:- [w0 w1 w2 w3 w4]= ',w)
for j in range(len(w)):
    for i in range(len(t_test)):

        x_T_test[i,j]=X_test[i]**j

y_pred = np.matmul(x_T_test,w)

#MEAN ABSOLUTE ERROR
MAE=0
for i in range(len(t_test)):

    MAE=MAE+np.abs(t_test[i]-y_pred[i])

print('Mean Absolute Error:- ',MAE/len(t_test))

#PLOTTING
plt.scatter(X_test, t_test,color='r',label='Target Value')

plt.scatter(X_test, y_pred,color='g',label='Predicted Value')

plt.title('Prediction of sine data using polynomial with regularizer lamda=1.8')

plt.xlabel("TESTING INPUTS---->")

plt.ylabel('TARGET/PREDICTED VALUES---->')

plt.legend(loc='upper right')

plt.show()