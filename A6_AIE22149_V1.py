from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import numpy.linalg as nl 

#Implements the step activation function
def step_activation_function(a):
    if a>=0:
        return 1
    else:
        return 0
    
#Implements the sigmoid function
def Sigmoid(x):
    return 1/(1+(math.e**(-x)))

#Implements the bipolar step function
def bipolar(a):
    if a>0:
        return 1
    elif a<0:
        return -1
    elif a==0:
        return 0
    
#Implements the ReLU function   
def ReLU(a):
    if a>0:
        return a
    else:
        return 0
    
def question1(input_list,target):
    w0=10 #bias
    w1=0.2 #Weight 1
    w2=-0.75 #Weight 2
    learning_rate=0.05 #alpha- learning rate
    epoch_errors=[]
    epochs=1000 #Maximum number of epochs
    predictions=[]
    for epoch in range(epochs):
        epoch_error=0
        y_in=0
        #Find the linear combination of the weights and inputs plus bias, give it to the activation function and calculate the error to adjust the weights.
        for i in range(len(input_list)):
            y_in+=(w1*input_list[i][0]+w2*input_list[i][1])+w0
            y_pred=step_activation_function(y_in)
            predictions.append(y_pred)
            error=target[i]-y_pred
            epoch_error+=error**2
            if error!=0:
                w0=w0+learning_rate*error
                w1=w1+(learning_rate*error*input_list[i][0])
                w2=w2+(learning_rate*error*input_list[i][1])
        epoch_errors.append(epoch_error)
        if epoch_error<0.002: #If error is less than threshold, plot the graph and exit.
            plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
            plt.xlabel('Epoches')
            plt.ylabel('Error')
            plt.title('Error v/s Epochs - Sigmoid function')
            plt.show()
            return epoch+1
        
    plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
    plt.xlabel('Epoches')
    plt.ylabel('Error')
    plt.title('Error v/s Epochs - ReLU function')
    plt.show()
    return None
    
#Implements the single layer perceptron but uses bipolar step function as activation function
def question2_bipolar(input_list,target):
    w0=10 #bias
    w1=0.2
    w2=-0.75
    learning_rate=0.05

    epoch_errors=[]
    epochs=1000
    predictions=[]
    for epoch in range(epochs):
        epoch_error=0
        y_in=0
        for i in range(len(input_list)):
            y_in+=(w1*input_list[i][0]+w2*input_list[i][1])+w0
            y_pred=bipolar(y_in)
            predictions.append(y_pred)
            error=target[i]-y_pred
            epoch_error+=error**2
            if error!=0:
                w0=w0+learning_rate*error
                w1=w1+(learning_rate*error*input_list[i][0])
                w2=w2+(learning_rate*error*input_list[i][1])
        epoch_errors.append(epoch_error)
        if epoch_error<0.002:
            plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
            plt.xlabel('Epoches')
            plt.ylabel('Error')
            plt.title('Error v/s Epochs - Sigmoid function')
            plt.show()
            return epoch+1
        
    plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
    plt.xlabel('Epoches')
    plt.ylabel('Error')
    plt.title('Error v/s Epochs - ReLU function')
    plt.show()
    return None

#Implements the single layer perceptron but uses sigmoid function as activation function    
def question2_sigmoid(input_list,target):
    w0=10 #bias
    w1=0.2
    w2=-0.75
    learning_rate=0.05
    
    epoch_errors=[]
    epochs=1000
    predictions=[]
    for epoch in range(epochs):
        epoch_error=0
        y_in=0
        for i in range(len(input_list)):
            y_in+=(w1*input_list[i][0]+w2*input_list[i][1])+w0
            y_pred=Sigmoid(y_in)
            predictions.append(y_pred)
            error=target[i]-y_pred
            epoch_error+=error**2
            if error!=0:
                w0=w0+learning_rate*error
                w1=w1+(learning_rate*error*input_list[i][0])
                w2=w2+(learning_rate*error*input_list[i][1])
        epoch_errors.append(epoch_error)
        if epoch_error<0.002:
            plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
            plt.xlabel('Epoches')
            plt.ylabel('Error')
            plt.title('Error v/s Epochs - Sigmoid function')
            plt.show()
            return epoch+1
            
    plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
    plt.xlabel('Epoches')
    plt.ylabel('Error')
    plt.title('Error v/s Epochs - Sigmoid function')
    plt.show()
    return None
  
#Implements the single layer perceptron but uses ReLU function as activation function 
def question2_ReLU(input_list,target):
    w0=10 #bias
    w1=0.2
    w2=-0.75
    learning_rate=0.05
    
    epoch_errors=[]
    epochs=1000
    predictions=[]
    for epoch in range(epochs):
        epoch_error=0
        y_in=0
        for i in range(len(input_list)):
            y_in+=(w1*input_list[i][0]+w2*input_list[i][1])+w0
            y_pred=ReLU(y_in)
            predictions.append(y_pred)
            error=target[i]-y_pred
            epoch_error+=error**2
            if error!=0:
                w0=w0+learning_rate*error
                w1=w1+(learning_rate*error*input_list[i][0])
                w2=w2+(learning_rate*error*input_list[i][1])
        epoch_errors.append(epoch_error)
        if epoch_error<0.002:
            plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
            plt.xlabel('Epoches')
            plt.ylabel('Error')
            plt.title('Error v/s Epochs - Sigmoid function')
            plt.show()
            return epoch+1
        
    plt.plot(range(1,len(epoch_errors)+1),epoch_errors)
    plt.xlabel('Epoches')
    plt.ylabel('Error')
    plt.title('Error v/s Epochs - ReLU function')
    plt.show()
    return None

#Uses the single layer perceptron with different learning rates. and plots it.
def question3(input_list,target):
    rates=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    w0=10
    w1=0.2
    w2=-0.75
    plot_data=[]
    for j in rates:
        learning_rate=j
        epoch_errors=[]
        epochs=1000
        predictions=[]
        for epoch in range(epochs):
            epoch_error=0
            y_in=0
            for i in range(len(input_list)):
                y_in+=(w1*input_list[i][0]+w2*input_list[i][1])+w0
                y_pred=ReLU(y_in)
                predictions.append(y_pred)
                error=target[i]-y_pred
                epoch_error+=error**2
                if error!=0:
                    w0=w0+learning_rate*error
                    w1=w1+(learning_rate*error*input_list[i][0])
                    w2=w2+(learning_rate*error*input_list[i][1])
            epoch_errors.append(epoch_error)
            if epoch_error<0.002:
                plot_data.append((learning_rate,epoch+1))
            else:
                plot_data.append((learning_rate,1000))
    
    x_vals=[x for x,_ in plot_data]
    y_vals=[y for _,y in plot_data]
    plt.scatter(x_vals,y_vals)
    plt.title("Learning rate v/s Number of iterations")
    plt.xlabel("Learning rate")
    plt.ylabel("No. of iterations")
    plt.show()
  
#Single layer perceptron on purchase data to check if it is a high value transaction.
def question5():
    df=pd.read_excel("Lab Session1 Data (1).xlsx")
    df=pd.DataFrame(df)
    df.dropna(axis=1,inplace=True)
    matrix=df.values
    a=[]
    for i in range(len(matrix)):
        for j in range(1,5):
            a.append(matrix[i][j])
    A=np.array(a)
    A=A.reshape(10,4)  #Matrix A
    target=np.array([])
    for i in range(len(matrix)):
      target=np.append(target,matrix[i][-1]) 
    bias=10 
    w1=0.2
    w2=0.35
    w3=0.4
    w4=0.5
    learning_rate=0.05
    epoch_errors=[]
    epochs=1000
    predictions=[]
    for epoch in range(epochs):
        epoch_error=0
        y_in=0
        for i in range(len(A)):
            y_in+=(w1*A[i][0]+w2*A[i][1]+w3*A[i][2]+w4*A[i][3])+bias
            y_pred=Sigmoid(y_in)
            error=target[i]-y_pred
            epoch_error+=error**2
            if error!=0:
                bias=bias+learning_rate*error
                w1=w1+(learning_rate*error*A[i][0])
                w2=w2+(learning_rate*error*A[i][1])
                w3=w3+(learning_rate*error*A[i][2])
                w4=w4+(learning_rate*error*A[i][3])
        epoch_errors.append(epoch_error)
        if epoch_error<0.002:
            
            break
    return predictions
          
    
def question6():
    df=pd.read_excel("Lab Session1 Data (1).xlsx")
    df=pd.DataFrame(df)
    df.dropna(axis=1,inplace=True)
    matrix=df.values
    a=[]
    for i in range(len(matrix)):
        for j in range(1,4):
            a.append(matrix[i][j])
    A=np.array(a)
    A=A.reshape(10,3)  #Matrix A
    C=np.array([])
    for i in range(len(matrix)):
      C=np.append(C,matrix[i][-2]) #Matrix C   
    pinv=np.linalg.pinv(A)
    cost=np.dot(pinv,C)
    return cost

#AND gate classified using MLPClassifier()    
def question10_And():
    X=[[0,0],[0,1],[1,0],[1,1]]
    And_Y=[0,0,0,1]
    Xor_Y=[0,1,1,0]
    clf=MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2,),activation='identity', max_iter=10000, random_state=1)
    clf.fit(X,And_Y)
    predictions=clf.predict(X)
    
    return predictions
#XOR gate classified using MLPClassifier()  
def question10_XOR():
    X=[[0,0],[0,1],[1,0],[1,1]]
    Xor_Y=[0,1,1,0]
    clf_xor=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(4,),activation='logistic',max_iter=10000,random_state=1) 
    clf_xor.fit(X,Xor_Y)
    predictions=clf_xor.predict(X)
    
    return predictions    

def main():
    op=int(input('''
                 Enter the option:
                 1. Perceptron using Step activation function for AND gate
                 2. Perceptron using Bi-polar step function, sigmoid and ReLU function.
                 3. Different learning rates
                 4. XOR gate
                 5. Purchase Data
                 6. Purchase data pseudo-inverse
                 8. A1 in XOR gate
                 10. AND and XOR gate using MLPClassifier()
                 '''))
    if op==1:
        X=[[0,0],[0,1],[1,0],[1,1]]
        target=[0,0,0,1]
        ans=question1(X,target)
        if ans!=None:
            print(f"Number of epoches is {ans}.")
    if op==2:
        X=[[0,0],[0,1],[1,0],[1,1]]
        target=[0,0,0,1]        
        ans1=question2_bipolar(X,target)
        if ans1!=None:
            print(f"Number of epoches with bipolar function is {ans1}.")
        ans2=question2_sigmoid(X,target)
        if ans2!=None:
            print(f"Number of epoches with sigmoid function is {ans2}.")
        ans3=question2_ReLU(X,target)
        if ans3!=None:
            print(f"Number of epoches with ReLU function is {ans3}.")
    if op==3:
        X=[[0,0],[0,1],[1,0],[1,1]]
        target=[0,0,0,1]
        question3(X,target)
    if op==4:
        X=[[0,0],[0,1],[1,0],[1,1]]
        target=[0,1,1,0]
        ans=question1(X,target)
        if ans!=None:
            print(f"Number of epoches is {ans}.")
        X=[[0,0],[0,1],[1,0],[1,1]]
        target=[0,1,1,0]        
        ans1=question2_bipolar(X,target)
        if ans1!=None:
            print(f"Number of epoches with bipolar function is {ans1}.")
        ans2=question2_sigmoid(X,target)
        if ans2!=None:
            print(f"Number of epoches with sigmoid function is {ans2}.")
        ans3=question2_ReLU(X,target)
        if ans3!=None:
            print(f"Number of epoches with ReLU function is {ans3}.") 
    if op==5:
        print(question5())
    if op==6:
        print(question6())       
    if op==8:
        X=[[0,0],[0,1],[1,0],[1,1]]
        target=[0,1,1,0]
        ans=question1(X,target)
        if ans!=None:
            print(f"Number of epoches is {ans}.")
    if op==10:
        print("AND Gate")
        print("Input:[0,0],[0,1],[1,0],[1,1] ")
        print("Target: [0,0,0,1] ")
        print(f"Predictions: {question10_And()}")
        
        print("XOR Gate")
        print("Input:[0,0],[0,1],[1,0],[1,1] ")
        print("Target: [0,1,1,0] ")
        print(f"Predictions: {question10_XOR()}")
        

