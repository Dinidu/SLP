import numpy as np
from numpy import genfromtxt

data_set = genfromtxt('input.csv', delimiter=',')
#print(data_set)

#split dataset in to input and output input is with 3 attributes
iR,oR = np.hsplit(data_set,np.array([4, ]))

#calculate dimentions of input and output vectors
print("Input and Output dimentions: ")
print(np.shape(iR)[1],np.shape(oR)[1])
di = np.shape(iR)
do = np.shape(oR)

#initialize w values for the given dataset

print("\nInitial weights")
wR = np.random.rand(di[1]+1,do[1])
print(wR)
print("\nShape of weights :")
print(np.shape(wR))

#calculate weighted inputs
wRt = np.transpose(wR)
bias = [1]

print(type(iR[0]))

iRi=np.append(iR[0],bias,axis=0)
print(iRi)
oRi=np.transpose(oR[0])

print(np.shape(wRt),np.shape([iRi]))

v=sum(np.dot(wRt,iRi))
print("\nCalculating V")
print("wRt ::\n"+str(wRt))
print("          ")
print("iRi ::\n"+str(iRi))

print("\nSum of weighted input [sum(wRt*iRi)]::\n"+str(v))

#activation function ::
y = np.sign(v)
print("output y ::\n"+str(y))

#calculate error ::
eR = y - oRi
print("Error ::\n")