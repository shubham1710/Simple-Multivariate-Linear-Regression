import numpy as np
import csv
a=np.genfromtxt('train_X.csv',delimiter=',')
b=np.genfromtxt('train_Y.csv',delimiter=',')
a=np.delete(a,0,0)
co=np.shape(a)[0]
ro=np.shape(a)[1]
x_bias = np.ones((co,1))
a = np.reshape(a,(co,ro))
a = np.append(x_bias,a,axis=1)
at=np.transpose(a)
at1=at.dot(a)
t1=np.linalg.inv(at1)
t2=at.dot(b)
th=t1.dot(t2)
th=th.reshape(1,ro+1)
b=b.reshape(co,1)
aw=np.empty(co, dtype=float)
for i in range (0,co):
    h=th[0][0]*a[i][0]+th[0][1]*a[i][1]+th[0][2]*a[i][2]+th[0][3]*a[i][3]+th[0][4]*a[i][4]  
    aw[i]=h
aw=aw.reshape(co,1)
np.savetxt("predicted_test_Y.csv", aw, delimiter=",")
