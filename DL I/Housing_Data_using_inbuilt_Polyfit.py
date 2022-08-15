##AKSHAY J
##21105012
import numpy as np
import pandas as pd

##Importing the dataset

data=pd.read_csv('housing.csv')
X=data.iloc[:,0:13].values
Y=data.iloc[:,13].values
##Splitting the data into testing and training
from sklearn.model_selection import train_test_split

X_train,X_test,t_train,t_test=train_test_split(X,Y,test_size=0.1,random_state=0)

#CALCULATON
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(X_train,t_train)
print('WEIGHT VALUES:',reg.coef_)
y_pred=reg.predict(X_test)

#MAE
MAE=0
for i in range(len(t_test)):

    MAE=MAE+np.abs(t_test[i]-y_pred[i])

print('Mean Absolute Error:- ',MAE/len(t_test))

#TESTING THE ACCUARCY USING ONE OF THE TEST VALUE
print('ACTUAL VALUE OF 47th TEST DATA: ',t_test[48])
print("PREDICTED VALUE OF 47th TEST DATA: ",y_pred[48])
