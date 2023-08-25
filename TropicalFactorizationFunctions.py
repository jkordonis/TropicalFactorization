import numpy as np
import math
from scipy import sparse
   

def trop_mat_mul(A,B):
  n_A=A.shape[0]
  m_A=A.shape[1]
  n_B=B.shape[0]
  m_B=B.shape[1]
  Prod=np.ones((n_A,m_B))*(-math.inf)
  if n_B!=m_A:
     raise Exception("Wrong dimensions")
  for i in range(n_A):
    for j in range(m_B):
      Prod[i,j]=np.max(A[i]+B[:,j])
  return Prod

def trop_mat_vec_mul(A,B):
  n_A=A.shape[0]
  Prod=np.ones(n_A)*(-math.inf)
  for i in range(n_A):
    Prod[i]=np.max(A[i]+B)

  return Prod


def trop_mat_mul_sparse_obs(A,B,RowInd,ColInd):
  n_A=A.shape[0]
  m_A=A.shape[1]
  n_B=B.shape[0]
  m_B=B.shape[1]
  NumbOfElements=RowInd.size
  Prod=np.ones(NumbOfElements)*(-math.inf)
  if (n_B!=m_A) or (RowInd.max()>n_A) or (ColInd.max()>m_B) :
     raise Exception("Wrong dimensions")
  for Ind in range(NumbOfElements):
    i=RowInd[Ind]
    j=ColInd[Ind]
    Prod[Ind]=np.max(A[i]+B[:,j])
  return sparse.coo_matrix((Prod,(RowInd,ColInd)))


def phase_1_w(C,D):
  n = C.shape[0]
  p = D.shape[1]
  r = D.shape[0]
  w = np.zeros((n,p,r))
  for i in range(n):
      for j in range(p):
            Add_C_D = C[i]+D[:,j]
            w[i,j,np.argmax(Add_C_D)]=1
  return w

def phase_1_w_soft(C,D,inv_tempr):
  n = C.shape[0]
  p = D.shape[1]
  r = D.shape[0]  
  w=np.zeros((n,p,r))
  for i in range(n):
      for j in range(p):
            Add_C_D = C[i]+D[:,j]
            w[i,j,:]=np.exp(Add_C_D*inv_tempr)/np.sum(np.exp(Add_C_D*inv_tempr))
  return w

def grad_c(C,D,B,w):
  n = C.shape[0]
  r = D.shape[0]  
  pertial_d=np.zeros((n,r))
  for i in range(n):
    for l in range(r):
      term_1 = np.sum(w[i,:,l])*C[i,l]
      term_2 = np.sum(w[i,:,l]*D[l,:])
      term_3 = np.sum(w[i,:,l]*B[i,:])
      pertial_d[i,l]=term_1+term_2-term_3
  return pertial_d


def grad_d(C,D,B,w):
  p = D.shape[1]
  r = D.shape[0]  
  pertial_d=np.zeros((r,p))
  for l in range(r):
    for j in range(p):
      term_1 = np.sum(w[:,j,l]*C[:,l])
      term_2 = np.sum(w[:,j,l]*D[l,j])
      term_3 = np.sum(w[:,j,l]*B[:,j])
      pertial_d[l,j]=term_1+term_2-term_3
  return pertial_d


def Factorization_Iteration(B,r,C0,D0,ITERATIONS,Speaking=False,DiminishingStepSize=True,DiminishingStepSizeConst=1000):
  C=C0   
  D=D0
  for iter in range(ITERATIONS):
    w=phase_1_w(C,D)
    AddConst = 1/(iter+DiminishingStepSizeConst)
    C=C-0.01*grad_c(C,D,B,w)+AddConst 
    D=D-0.01*grad_d(C,D,B,w)+AddConst 
    if np.mod(iter*100,ITERATIONS)==0:
        if Speaking:
            print(iter*100/ITERATIONS,'%, Error: ',np.linalg.norm(B-trop_mat_mul(C,D)))
  return C,D

