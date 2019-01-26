from numpy import *

error_bias=array([])
error_zero_bias=array([])

for i in range(10):
    #data input
    file_name='data/dataset'+str(i)+'.npz'
    dataset=load(file_name)
    train_x=dataset['train_x']
    train_y=dataset['train_y']
    test_x=dataset['test_x']
    test_y=dataset['test_y']
    #solve with bias
    w=dot(dot(linalg.inv(dot(train_x.T,train_x)),train_x.T),train_y)
    test_predict=dot(test_x,w)
    error_bias=append(error_bias,mean(abs(test_predict-test_y)))
    #solve without bias
    train_x=delete(train_x,-1,1)
    test_x=delete(test_x,-1,1)
    w=dot(dot(linalg.inv(dot(train_x.T,train_x)),train_x.T),train_y)
    test_predict=dot(test_x,w)
    error_zero_bias=append(error_zero_bias,mean(abs(test_predict-test_y)))

print("Errors with bias:")
print(error_bias)
print("Average error with bias:",mean(error_bias))
print("Errors with zero bias:")
print(error_zero_bias)
print("Average error with zero bias:",mean(error_zero_bias))
