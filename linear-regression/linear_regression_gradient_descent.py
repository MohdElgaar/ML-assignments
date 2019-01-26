from pylab import *
from numpy import *

eta=0.0009
threshold=0.01
errors=array([])

def gradient_decent(x):
    train_x=x['train_x']
    train_y=x['train_y']
    test_x=x['test_x']
    test_y=x['test_y']
    #initialize variables
    #w=random.normal(0,1,train_x.shape[1])
    w=array(list(range(train_x.shape[1])),dtype='float64')
    new_e=10000
    old_e=20000
    #for graph
    error_progression=list()
    analytic = dot(dot(linalg.inv(dot(train_x.T,train_x)),train_x.T),train_y)
    while(old_e-new_e>threshold):
    #for i in range(20):  #for graph
        grad=dot(dot(w.T,train_x.T),train_x)-dot(train_y.T,train_x)
        w-=eta*grad
        cost=dot(train_x,w)
        error=linalg.norm(cost-train_y)
        error_progression.append(error)
        old_e=new_e
        new_e=error_progression[-1]
   #for graph
   # plot(error_progression)
   # axhline(linalg.norm((dot(train_x,analytic))-train_y))
   # show()
    return mean(abs(dot(test_x,w)-test_y))

for i in range(10):
    file_name='data/dataset'+str(i)+'.npz'
    x=load(file_name)
    e=gradient_decent(x)
    errors=append(errors,e)

print("Errors:")
print(errors)
print("Average error",mean(errors))
