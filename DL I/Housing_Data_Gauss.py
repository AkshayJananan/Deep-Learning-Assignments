##AKSHAY J
##21105012

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

##Importing the dataset
data=pd.read_csv('housing.csv')

#PREPROCESSING DATA USING MINIMAX SCALAR
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaled=scaler.fit_transform(data)
X=scaled[:,0:13]
Y=scaled[:,13]

##Splitting the data into testing and training
from sklearn.model_selection import train_test_split

X_train,X_test,t_train,t_test=train_test_split(X,Y,test_size=0.1,random_state=0)

#INITIALISING
ones1=np.ones(len(t_train))
ones2=np.ones(len(t_test))
zeros=np.zeros([1,13])
x_T=X_train.copy()
x_T_test=X_test.copy()
w=np.ones(4)
phi_T = np.ones([len(t_train), len(w)])
phi_test = np.ones([len(t_test), len(w)])
eta=10**-3

#SETTING RADIAL GAUSS FUNCTION
def gauss(x,mu,variance):
    sum=0
    for i in range(len(x)):
        sum=sum+((x[i]-mu[i])**2)
    q=np.exp(-(sum)/(2*variance)) ##exp(-(||xj-muj||)/2*variance))
    return q

from sklearn.cluster import MiniBatchKMeans

kmeans=MiniBatchKMeans(n_clusters=3,random_state=0,batch_size=6)
#Setting the mean and variance of training data
kmeans.fit_transform(x_T)
km_labels1=kmeans.predict(x_T)
cluster1=0
cluster2=0
cluster3=0
variance1=np.zeros(4)
for i in range(len(x_T)):
    if km_labels1[i]==0:
        cluster1=np.append(cluster1,x_T[i])
    elif km_labels1[i]==1:
        cluster2=np.append(cluster2,x_T[i])
    elif km_labels1[i]==2:
        cluster3=np.append(cluster3,x_T[i])
variance1[1]=np.var(cluster1)
variance1[2]=np.var(cluster2)
variance1[3]=np.var(cluster3)
mu1=kmeans.cluster_centers_
mu1=np.insert(mu1,0,zeros,axis=0)

#Setting the mean and variance of testing data
kmeans.fit_transform(x_T_test)
km_labels2=kmeans.predict(x_T_test)
cluster4=0
cluster5=0
cluster6=0
variance2=np.zeros(4)
for i in range(len(x_T_test)):
    if km_labels2[i]==0:
        cluster4=np.append(cluster4,x_T_test[i])
    elif km_labels2[i]==1:
        cluster5=np.append(cluster5,x_T_test[i])
    elif km_labels2[i]==2:
        cluster6=np.append(cluster6,x_T_test[i])
variance2[1]=np.var(cluster4)
variance2[2]=np.var(cluster5)
variance2[3]=np.var(cluster6)
mu2=kmeans.cluster_centers_
mu2=np.insert(mu2,0,zeros,axis=0)

#CALCULATION
for j in range(1,4):
    for i in range(len(x_T)):
        phi_T[:,0]=1
        phi_T[i,j]=gauss(x_T[i,:],mu1[j,:],variance1[j])

for k in range(1000):
    y = np.matmul(phi_T, w)

    der = np.matmul(np.transpose(phi_T), np.subtract(y, t_train))

    w = np.subtract(w, eta * der)

print('Weight values:- [w0 w1 w2 w3]= ',w)
for j in range(1,4):
    for i in range(len(x_T_test)):
        phi_test[:, 0] = 1
        phi_test[i,j]=gauss(x_T_test[i,:],mu2[j,:],variance2[j])
y_pred = np.matmul(phi_test,w)

#MEAN ABSOLUTE ERROR
MAE=0
for i in range(len(t_test)):

    MAE=MAE+np.abs(t_test[i]-y_pred[i])

print('Mean Absolute Error:- ',MAE/len(t_test))

#PLOTTING
plt.scatter(X_test[:,6], t_test,color='r',label='Target Value')

plt.scatter(X_test[:,6], y_pred,color='g',label='Predicted Value')

plt.title('Prediction of Housing data using Radial Basis Gauss function plotted with 5th input')

plt.xlabel("5th INPUTS IN THE HOUSING DATA---->")

plt.ylabel('TARGET/PREDICTED VALUES---->')

plt.legend(loc='upper right')

plt.show()

