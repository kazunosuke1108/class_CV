import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
a=np.insert(a,0,np.zeros((3,1)),axis=0)
print(a)