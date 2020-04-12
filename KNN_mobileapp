import numpy as np 
import pandas as pd 
df=pd.read_csv("PML.csv")

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
X=df[['sup_devices.num']]
Y=df[['ipadSc_urls.num']]
X_train,X_test,y_train,y_test= train_test_split(X, Y, test_size = 0.2, random_state=42) 
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
p1=knn.predict([['37']])
p1[0]
