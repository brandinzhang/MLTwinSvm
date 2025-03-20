import numpy as np
from cvxopt import matrix, solvers

def solve(R,S,c1,Epsi1):
    # temp = (S^TS)^-1R^T
    m1 = R.shape[0]
    temp = np.linalg.inv(S.T@S+Epsi1*np.eye(S.shape[1]))@R.T
    P = matrix(R@temp,tc = 'd')
    q = matrix(-np.ones(m1),tc = 'd')

    zeros = np.zeros(m1)
    # 修改 h 的生成方式，使其成为单列矩阵
    h = matrix(np.hstack((zeros, 1.0*c1*np.ones(m1))).reshape(-1, 1), tc='d')

    G1 = -np.eye(m1)
    G2 = np.eye(m1)
    G = matrix(np.vstack((G1,G2)),tc = 'd')

    A = None
    b = None

    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h,A,b)
    alpha = np.array(sol['x'])
    z = -temp@alpha
    # return u1,b1,u是二维列向量
    u1 = z[:len(z)-1].flatten()
    b1 = z[len(z)-1]

    return u1,b1,alpha
