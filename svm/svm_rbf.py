import pylab as pl
import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress']  =  False
test_sets = np.load('data/test.npz')
test_x = test_sets['inputs']
test_y = test_sets['targets']
Cs = np.array([9])
sigmas = np.array([4])
#Cs = np.arange(0,1,0.1)[1:]
#sigmas = np.arange(0,1,0.1)[1:]
#data = np.array([])
sigma = None

gaussian_kernel  =  lambda x,y: np.exp(-np.linalg.norm(x-y)**2/(2*(sigma**2)))

def gaussian_gram(X,Y):
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            K[i,j] = gaussian_kernel(X[i,:],Y[j,:])
    return K

def predict(sample_x, sv_ay, sv_x):
    prediction_K = gaussian_gram(sample_x,sv_x)
    return np.dot(sv_ay,prediction_K.T)

def solver():
    #global data
    global sigma
    train_sets = np.load('data/trainval.npy')
    train_inputs = train_sets[0]
    train_targets = train_sets[1]
    valid_errors = np.array([])
    train_errors = np.array([])
    for C in Cs:
        for sigma in sigmas:
            #ve_progression = np.array([])
            te_progression = np.array([])
            #for i in range(5):
            #train_x = np.vstack([train_inputs[j] for j in range(5) if j!=i])
            #train_y = np.concatenate([train_targets[j] for j in range(5) if j!=i])
            #valid_x = np.array(train_inputs[i])
            #valid_y = np.array(np.array(train_targets[i]))
            train_x = np.vstack(train_inputs)
            train_y = np.concatenate(train_targets)
            n = train_y.size
            K = gaussian_gram(train_x,train_x)
            P = cvxopt.matrix(np.outer(train_y,train_y)*K)
            q = cvxopt.matrix(-np.ones(n))
            G = cvxopt.matrix(np.vstack((np.diag(-np.ones(n)),np.identity(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n),np.full(n,C))))
            A = cvxopt.matrix(train_y,(1,n))
            b = cvxopt.matrix(0.0)
            solution = np.ravel(cvxopt.solvers.qp(P,q,G,h,A,b)['x'])
            sv_indices = solution>1e-4
            sv_x = train_x[sv_indices]
            sv_y = train_y[sv_indices]
            sv_a = solution[sv_indices]
            sv_ay = sv_a*sv_y
            marginal_ind = np.where((solution>0.1*C) & (solution<0.9*C))[0][0]
            b = train_y[marginal_ind]-np.dot(solution*train_y, K.T[marginal_ind])
            #v_prediction = np.sign(predict(valid_x,sv_ay,sv_x)+b)
            t_prediction = np.sign(predict(train_x,sv_ay,sv_x)+b)
            #v_difference = np.where(v_prediction==valid_y,1,0)
            t_difference = np.where(t_prediction==train_y,1,0)
            #ve_progression = np.append(ve_progression,1-sum(v_difference)/valid_y.size)
            te_progression = np.append(te_progression,1-sum(t_difference)/train_y.size)
            #valid_errors = np.append(valid_errors,np.mean(ve_progression))
            train_errors = np.append(train_errors,np.mean(te_progression))
            #data = np.append(data,np.array([C,sigma,valid_errors[-1]]))
           # print(C,sigma)
           # print(np.mean(ve_progression))
           # print(np.mean(te_progression))
           # print('')
    #pl.bar(Cs-0.1,valid_errors,0.15,label = 'validation')
    #pl.bar(Cs+0.1,train_errors,0.15,label = 'training')
    #pl.bar(sigmas-0.1,valid_errors,0.15,label = 'validation')
    #pl.bar(sigmas+0.1,train_errors,0.15,label = 'training')
    #pl.legend()
    #pl.show()
    return sv_ay, sv_x, b


sv_ay, sv_x, b = solver()
#np.savetxt('parameters2.csv', data, delimiter='\t')
test_predict = np.sign(predict(test_x, sv_ay, sv_x)+b)
prediction_difference = np.where(test_predict==test_y, 1, 0)
error = 1-np.sum(prediction_difference)/test_y.size
print(error)
