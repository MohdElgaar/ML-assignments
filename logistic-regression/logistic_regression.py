import pylab as pl
import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
test_sets=np.load('data/test.npz')
test_x=test_sets['inputs']
test_x=np.hstack((test_x,np.ones((test_x.shape[0],1))))
test_y=test_sets['targets']
eta=0.0009
threshold=0.01

def predict(inputs,beta):
    return np.where(np.dot(inputs,beta)>=0, 1, -1)

def train():
    train_sets=np.load('data/trainval.npy')
    train_inputs=train_sets[0]
    train_inputs=[np.hstack((x,np.ones((x.shape[0],1)))) for x in train_inputs]
    train_targets=train_sets[1]
    final_errors=np.array([])
    train_x=np.vstack(train_inputs)
    train_y=np.concatenate(train_targets)
    beta=np.random.normal(size=124)
    old_e=10000
    new_e=9000
    errors=np.array([])
    while(old_e-new_e>threshold):
        grad=-np.dot((np.where(train_y==-1,0,train_y)-sigmoid(np.dot(train_x,beta))), train_x)
        beta-=eta*grad
        t_prediction=predict(train_x,beta)
        old_e=new_e
        new_e=np.sum(np.log(1+np.exp(-np.dot(beta,train_x.T)*train_y)))
        errors=np.append(errors,new_e)
    pl.plot(errors)
    print("Training Error:", 1-np.sum(np.where(predict(train_x,beta)==train_y,1,0))/train_y.size)
    return beta

beta=train()
test_error=1-np.sum(np.where(predict(test_x,beta)==test_y,1,0))/test_y.size
print("Test Error:",test_error)
pl.show()
