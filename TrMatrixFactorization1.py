import numpy as np
import math

import matplotlib.pyplot as plt


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
  n_B=B.shape[0]
  m_A=A.shape[1]
  Prod=np.ones(n_A)*(-math.inf)
  for i in range(n_A):
    Prod[i]=np.max(A[i]+B)

  return Prod

n=100
p=120
r=5
C=np.random.rand(n,r)
D=np.random.rand(r,p)
C_or=C
D_or=D

B=trop_mat_mul(C,D)+np.random.rand(n,p)*0.01
np.linalg.norm(trop_mat_mul(C,D)-B)

def phase_1_w(C,D):
  w=np.zeros((n,p,r))
  for i in range(n):
      for j in range(p):
            Add_C_D = C[i]+D[:,j]
            w[i,j,np.argmax(Add_C_D)]=1
  return w

def phase_1_w_soft(C,D,inv_tempr):
  w=np.zeros((n,p,r))
  for i in range(n):
      for j in range(p):
            Add_C_D = C[i]+D[:,j]
            w[i,j,:]=np.exp(Add_C_D*inv_tempr)/np.sum(np.exp(Add_C_D*inv_tempr))
  return w

def grad_c(C,D,B,w):
  pertial_d=np.zeros((n,r))
  for i in range(n):
    for l in range(r):
      term_1 = np.sum(w[i,:,l])*C[i,l]
      term_2 = np.sum(w[i,:,l]*D[l,:])
      term_3 = np.sum(w[i,:,l]*B[i,:])
      pertial_d[i,l]=term_1+term_2-term_3
  return pertial_d


def grad_d(C,D,B,w):
  pertial_d=np.zeros((r,p))
  for l in range(r):
    for j in range(p):
      term_1 = np.sum(w[:,j,l]*C[:,l])
      term_2 = np.sum(w[:,j,l]*D[l,j])
      term_3 = np.sum(w[:,j,l]*B[:,j])
      pertial_d[l,j]=term_1+term_2-term_3
  return pertial_d


## Iteration ##

C.shape

C0=np.random.rand(C.shape[0],C.shape[1])
D0=np.random.rand(D.shape[0],D.shape[1])

C=C0  #C_or+np.random.randn(C.shape[0],C.shape[1])
D=D0  # D_or+np.random.randn(D.shape[0],D.shape[1])
print(np.linalg.norm(B-trop_mat_mul(C0,D0)))

iter_cnt=0

Err_plt = np.zeros(10000)

for iter in range(50):
  for gd_iter in range(200):
    w=phase_1_w(C,D)
    #w=phase_1_w_soft(C,D,10)    
    C=C-0.01*grad_c(C,D,B,w)+0.000
    D=D-0.01*grad_d(C,D,B,w)+0.000
    Err_plt[iter_cnt]=np.linalg.norm(B-trop_mat_mul(C,D))
    iter_cnt+=1
  print(np.linalg.norm(B-trop_mat_mul(C,D)))

plt.plot(Err_plt, label = "Original")

C=C0  #C_or+np.random.randn(C.shape[0],C.shape[1])
D=D0  # D_or+np.random.randn(D.shape[0],D.shape[1])


iter_cnt=0

Err_plt = np.zeros(10000)

for iter in range(50):
  for gd_iter in range(200):
    w=phase_1_w(C,D)
    #w=phase_1_w_soft(C,D,10)    
    C=C-0.01*grad_c(C,D,B,w)+0.0002
    D=D-0.01*grad_d(C,D,B,w)+0.0002
    Err_plt[iter_cnt]=np.linalg.norm(B-trop_mat_mul(C,D))
    iter_cnt+=1
  print(np.linalg.norm(B-trop_mat_mul(C,D)))

plt.plot(Err_plt, label = "Impr.2 ε=0.0002")

C=C0   
D=D0

iter_cnt=0

Err_plt = np.zeros(10000)

for iter in range(50):
  for gd_iter in range(200):
    w=phase_1_w(C,D)
    #w=phase_1_w_soft(C,D,10)    
    C=C-0.01*grad_c(C,D,B,w)+0.0005
    D=D-0.01*grad_d(C,D,B,w)+0.0005
    Err_plt[iter_cnt]=np.linalg.norm(B-trop_mat_mul(C,D))
    iter_cnt+=1
  print(np.linalg.norm(B-trop_mat_mul(C,D)))

plt.plot(Err_plt, label = "Impr.2 ε=0.0005")

C=C0   
D=D0

iter_cnt=0

Err_plt = np.zeros(10000)

for iter in range(50):
  for gd_iter in range(200):
    w=phase_1_w(C,D)
    #w=phase_1_w_soft(C,D,10)    
    C=C-0.01*grad_c(C,D,B,w)+0.001
    D=D-0.01*grad_d(C,D,B,w)+0.001
    Err_plt[iter_cnt]=np.linalg.norm(B-trop_mat_mul(C,D))
    iter_cnt+=1
  print(np.linalg.norm(B-trop_mat_mul(C,D)))

plt.plot(Err_plt, label = "Impr.2 ε=0.001")


C=C0   
D=D0

iter_cnt=0

Err_plt = np.zeros(10000)

for iter in range(50):
  for gd_iter in range(200):
    w=phase_1_w(C,D)
    #w=phase_1_w_soft(C,D,10)    
    C=C-0.01*grad_c(C,D,B,w)+1/(iter_cnt+1000)
    D=D-0.01*grad_d(C,D,B,w)+1/(iter_cnt+1000)
    Err_plt[iter_cnt]=np.linalg.norm(B-trop_mat_mul(C,D))
    iter_cnt+=1
  print(np.linalg.norm(B-trop_mat_mul(C,D)))

plt.plot(Err_plt, label = "Impr.2 ε=1/(k+1000)")

plt.xlabel("Timesteps")
plt.ylabel("Norm of the error")



plt.legend()
plt.grid()


plt.savefig('TropicalMatrixFactorization1.pdf')
plt.show()
