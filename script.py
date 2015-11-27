import numpy as np
import math
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from numpy.linalg import inv
import scipy.io
import matplotlib.pyplot as plt
import pickle
import os

path="E:\Spring 2015\Machine Learning\PA2"
os.chdir(path)

def ldaLearn(X,y):
    means1 = np.zeros([5,2])
    covmat = np.zeros([X.shape[1],X.shape[1]])
    for a in range(1,6):
        count =0
        for b in range(0,y.shape[0]):
            if(y[b][0]==float(a)):
                count = count + 1
                means1[a-1][0] = means1[a-1][0] + X[b][0]
                means1[a-1][1] = means1[a-1][1] + X[b][1]
        means1[a-1][0] = means1[a-1][0]/count
        means1[a-1][1] = means1[a-1][1]/count
    means = np.transpose(means1)
    
    for c in range(1,6):
        for d in range(0,y.shape[0]):
            if(y[d][0]==float(c)):
                m = X[d].reshape(2,1)
                n = means1[c-1].reshape(2,1)
                covmat = covmat + np.dot((m-n),np.transpose(m-n)) 
               
    return means,covmat

def qdaLearn(X,y):
    means1 = np.zeros([5,2])
    covmat1 = np.zeros([X.shape[1],X.shape[1]])
    covmat2 = np.zeros([X.shape[1],X.shape[1]])
    covmat3 = np.zeros([X.shape[1],X.shape[1]])
    covmat4 = np.zeros([X.shape[1],X.shape[1]])
    covmat5 = np.zeros([X.shape[1],X.shape[1]])
    for a in range(1,6):
        count =0
        for b in range(0,y.shape[0]):
            if(y[b][0]==float(a)):
                count = count + 1
                means1[a-1][0] = means1[a-1][0] + X[b][0]
                means1[a-1][1] = means1[a-1][1] + X[b][1]
        means1[a-1][0] = means1[a-1][0]/count
        means1[a-1][1] = means1[a-1][1]/count
    means = np.transpose(means1)
    for c in range(1,6):
        for d in range(0,y.shape[0]):
            if(y[d][0]==float(c)):
                m = X[d].reshape(2,1)
                n = means1[c-1].reshape(2,1) 
                if(c==1):
                    covmat1 = covmat1 + np.dot((m-n),np.transpose(m-n))
                if(c==2):
                    covmat2 = covmat2 + np.dot((m-n),np.transpose(m-n))
                if(c==3):
                    covmat3 = covmat3 + np.dot((m-n),np.transpose(m-n)) 
                if(c==4):
                    covmat4 = covmat4 + np.dot((m-n),np.transpose(m-n))
                if(c==5):
                    covmat5 = covmat5 + np.dot((m-n),np.transpose(m-n)) 
    

    covmats = [covmat1, covmat2, covmat3, covmat4, covmat5]
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    means1 = np.transpose(means)
    temp = sqrt(np.linalg.det(covmat))
    acc =0
    for a in range(0,100):
        final = np.zeros([5,1])
        for i in range(0,5):
            temp1 = np.dot((Xtest[a]-means1[i]),( np.linalg.inv(covmat)))
            temp2 = np.transpose(Xtest[a]-means1[i])
            temp3 = -1/2*np.dot(temp1,temp2)
            b = (1/(sqrt(2*22/7)*temp))*(math.exp(temp3))
            final[i][0] = b
        if(float(final.argmax(axis=0)+1) == ytest[a][0]):
            acc = acc + 1; 
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    means1 = np.transpose(means)
    acc =0
    for a in range(0,100):
        final = np.zeros([5,1])
        for i in range(0,5):
            temp = sqrt(np.linalg.det(covmats[i]))
            temp1 = np.dot((Xtest[a]-means1[i]),(np.linalg.inv(covmats[i])))
            temp2 = np.transpose(Xtest[a]-means1[i])
            temp3 = -1/2*np.dot(temp1,temp2)
            b = (1/(sqrt(2*22/7)*temp))*(math.exp(temp3))
            final[i][0] = b
        if(float(final.argmax(axis=0)+1) == ytest[a][0]):
            acc = acc + 1; 
    return acc


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 
    w=np.array([]) 
    w=np.dot(inv((np.dot(X.T,X))),np.dot(X.T,y))
                                                
                                                 
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD  
    a=lambd * np.identity(X.shape[1])
    b=a *X.shape[0]
    c=np.dot(X.T,X)
    d=np.add(b,c)
    e=np.dot(X.T,y)
    w=np.dot(inv(d),e)                                              
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD  
       
        rmse=np.dot((ytest - np.dot(Xtest,w)).T,(ytest - np.dot(Xtest,w)))              
        rmse=np.sqrt(rmse)/Xtest.shape[0]
        return rmse 
        

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
    p=np.add(p,1)
    Xd = np.empty([x.shape[0],p]) 
    for i in range(x.shape[0]): 
        #row=x[i,:]
        #ans=row[0]
        ans = x[i]
        for j in range(p): 
            temp=math.pow(ans,j)
            Xd[i][j]=temp
            
    return Xd

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    w = np.reshape(w, (65,1)) 
    b = (lambd* np.dot(w.T, w))/2;
    a1 =  np.subtract(y, np.array(np.dot(X,w)));
    a = (np.dot(a1.T,a1))/(2*X.shape[0]);
    error = a + b;
    error = np.sum(error);
    p = np.dot(y.T, X);
    q = np.dot ( w.T, np.dot(X.T,X))
    r = np.multiply(lambd, w);
    
    error_grad = np.transpose((q - p)/X.shape[0]) + r;
    error_grad = np.squeeze(np.asarray(error_grad));
    
    
                                               
    return error, error_grad

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))
   

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
#add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
  
w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
# mle_train  = testOLERegression(w,X,y)
w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
# mle_train_i  = testOLERegression(w_i,X_i,y)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
# print('RMSE without intercept / train '+str(mle_train))
# print('RMSE with intercept / train '+str(mle_train_i))
# 
# # Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
plt.show()

# # Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()


# 
# # Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()