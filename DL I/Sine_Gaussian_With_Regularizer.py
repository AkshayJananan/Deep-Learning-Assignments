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

#INITIALISING
w = np.ones(5)
y = np.zeros(len(t_train))
y_pred = np.zeros(len(t_test))
eta = 10 **-3
phi_T = np.zeros([len(t_train), len(w)])
phi_test = np.zeros([len(t_test), len(w)])
lam=1.8

#SETTING THE GAUSSIAN FUNCTION
def gauss(x,mu):
    q=np.exp(-(x-mu)**2/(2*0.2**2)) #sd=0.2
    return(q)
mu=[0,0.2,0.4,0.6,0.8]

#CALCULATION
for j in range(1,len(w)):
    for i in range(len(t_train)):
        phi_T[i,0]=1

        phi_T[i,j]=gauss(X_train[i],mu[j])
for k in range(1000):
    y = np.matmul(phi_T, w)

    der =np.add(np.matmul(np.transpose(phi_T), np.subtract(y, t_train)),lam*w)

    w = np.subtract(w, eta * der)

print('Weight values:- [w0 w1 w2 w3 w4]= ',w)

for j in range(1,len(w)):
    for i in range(len(t_test)):
        phi_test[i,0]=1

        phi_test[i,j]=gauss(X_test[i],mu[j])

y_pred = np.matmul(phi_test,w)

#MEAN ABSOLUTE ERROR
MAE=0
for i in range(len(t_test)):

    MAE=MAE+np.abs(t_test[i]-y_pred[i])

print('Mean Absolute Error:- ',MAE/len(X_test))

#PLOTTING
plt.scatter(X_test, t_test,color='r',label='Target Value')

plt.scatter(X_test, y_pred,color='g',label='Predicted Value')

plt.title('Prediction of sine data using Gaussian with regularizer lambda=1.8')

plt.xlabel("TESTING INPUTS---->")

plt.ylabel('TARGET/PREDICTED VALUES---->')

plt.legend(loc='upper right')

plt.show()


