from pylab import *
from numpy import *

eta=0.0017
threshold=0.00000000001
errors=array([])
iterations=array([])

for i in range(10):
    file_name='data/dataset'+str(i)+".npz"
    x=load(file_name)
    error_progression=array([])
    old_e=10000
    new_e=1000
    train_x=x['train_x']
    train_y=x['train_y']
    test_x=x['test_x']
    test_y=x['test_y']
    analytic = dot(dot(linalg.inv(dot(train_x.T,train_x)),train_x.T),train_y)
    j=0
    w=linspace(-1,1,train_x.shape[1])
    while(old_e-new_e>threshold):
        w[j]-=eta*2*dot(dot(train_x,w)-train_y, train_x[:,j])
        error_progression=append(error_progression, linalg.norm(dot(train_x,w)-train_y))
        j+=1
        j=j%len(w)
        old_e=new_e
        new_e=error_progression[-1]
    errors=append(errors,mean(abs(dot(test_x,w)-test_y)))
    iterations=append(iterations,size(error_progression))
print("Errors")
print(errors)
print("Average error",mean(errors))
print("Average iterations",mean(iterations))
#axhline(linalg.norm((dot(train_x,analytic))-train_y))
#plot(error_progression)
#show()
