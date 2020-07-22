import numpy as np


a = np.array([[1,2],[34,2]])

b = np.ones([2,1])

c = np.concatenate((a,b),axis= 1)

print(a,'\n',b,"\n",c)