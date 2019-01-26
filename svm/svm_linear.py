import pylab as pl
import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False
test_sets=np.load('data/test.npz')
test_x=test_sets['inputs']
test_y=test_sets['targets']
Cs=np.array([0.245])
#Cs=np.arange(0.245,0.255,0.001)
#Cs=np.linspace(0.245,2.1,5)

def linear_kernel(X):
    return np.dot(X,X.T)

def solver():
    train_sets=np.load('data/trainval.npy')
    train_inputs=train_sets[0]
    train_targets=train_sets[1]
    valid_errors=np.array([])
    train_errors=np.array([])
    for C in Cs:
        ve_progression=np.array([])
        te_progression=np.array([])
        #for i in range(5):
        #train_x=np.vstack([train_inputs[j] for j in range(5) if j!=i])
        #train_y=np.concatenate([train_targets[j] for j in range(5) if j!=i])
        #valid_x=np.array(train_inputs[i])
        #valid_y=np.array(np.array(train_targets[i]))
        train_x=np.vstack(train_inputs)
        train_y=np.concatenate(train_targets)
        n=train_y.size
        K=linear_kernel(train_x)
        P=cvxopt.matrix(np.outer(train_y,train_y)*K)
        q=cvxopt.matrix(-np.ones(n))
        G=cvxopt.matrix(np.vstack((np.diag(-np.ones(n)),np.identity(n))))
        h=cvxopt.matrix(np.hstack((np.zeros(n),np.full(n,C))))
        A=cvxopt.matrix(train_y,(1,n))
        b=cvxopt.matrix(0.0)
        solution=np.ravel(cvxopt.solvers.qp(P,q,G,h,A,b)['x'])
        sv_indices=solution>1e-4
        sv_x=train_x[sv_indices]
        sv_y=train_y[sv_indices]
        sv_a=solution[sv_indices]
        w=np.dot(sv_y*sv_a, sv_x)
        marginal_ind=np.where((solution>0.1*C) & (solution<0.9*C))[0][0]
        b=train_y[marginal_ind]-np.dot(solution*train_y, K.T[marginal_ind])
        #v_prediction=np.sign(np.dot(w,valid_x.T)+b)
        t_prediction=np.sign(np.dot(w,train_x.T)+b)
        #v_difference=np.where(v_prediction==valid_y,1,0)
        t_difference=np.where(t_prediction==train_y,1,0)
        #ve_progression=np.append(ve_progression,1-sum(v_difference)/valid_y.size)
        te_progression=np.append(te_progression,1-sum(t_difference)/train_y.size)
        #valid_errors=np.append(valid_errors,np.mean(ve_progression))
        train_errors=np.append(train_errors,np.mean(te_progression))
    #pl.bar(Cs-0.05,valid_errors,0.07,label='validation')
    #pl.bar(Cs+0.05,train_errors,0.07,label='training')
    #pl.legend()
    #pl.show()
    return (w,b)


w,b = solver()
test_prediction=np.sign(np.dot(w,test_x.T)+b)
test_difference=np.where(test_prediction==test_y,1,0)
print(1-(sum(test_difference)/test_y.size))
