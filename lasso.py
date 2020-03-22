import cvxpy
import torch
import numpy as np 
from sklearn.linear_model import Lasso

def basis_pursuit(n, A, y, exact, e):
    P = A.numpy()
    q = y.numpy().flatten().astype(np.double) 
    x = cvxpy.Variable(n)
    obj = cvxpy.Minimize(cvxpy.norm(x,1))
    if exact:
        const = [P * x == q]
    else:
        const = [cvxpy.norm(P * x - q) <= e]
    prob = cvxpy.Problem(obj,const)
    result = prob.solve(verbose=False)#,eps_abs=1e-4,eps_rel=1e-4)
    return x.value

def generate_bp(m, n, x_star, exact=True, e=0.1):
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0/m]))
    A = normal.sample((m,n)).squeeze()
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.0])) # torch.tensor([0.1/np.sqrt(m)
    noise = normal.sample((m,))
    y = torch.mm(A,x_star) + noise

    x_hat = basis_pursuit(n, A, y, exact, e)
    return np.maximum(np.minimum(x_hat, 1), 0)



def generate_lasso(m,n,x_star, alpha):
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0/m]))
    A = normal.sample((m,n)).squeeze()
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.0]))
    noise = normal.sample((m,))
    y = torch.mm(A,x_star) + noise
    return solve_lasso(A,y, alpha)
    
def solve_lasso(A, y, alpha):
    A = A.numpy()
    y = y.numpy().flatten().astype(np.double)
    lasso_est = Lasso(alpha=alpha,max_iter=2000)
    lasso_est.fit(A, y)
    x_hat = lasso_est.coef_
    x_hat = np.reshape(x_hat, [-1])
    #print("max value: {}, min value: {}".format(x_hat.max(),x_hat.min()))
    x_hat = np.maximum(np.minimum(x_hat, 1), 0)
    
    return x_hat