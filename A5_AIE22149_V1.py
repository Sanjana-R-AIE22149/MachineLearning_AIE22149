import numpy as np
import pandas as pd
from numpy import linalg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

#Reads Styles excel sheet, extracts only certain columns and two classes, applies knn classifer and finds confusion matrix, recall, precision and f1 scores.
def question_A1():
    df=pd.read_excel('Styles.xlsx')
    columns_to_extract=['gender','masterCategory','subCategory','articleType','season','usage']
    new_df=df[columns_to_extract]
    class_men=new_df[new_df['gender']=='Men']
    class_women=new_df[new_df['gender']=='Women']
    updated_df=class_men._append(class_women,ignore_index=True)
    
    label_encoder=LabelEncoder()
    for column in updated_df.columns:
        if updated_df[column].dtype=='object':
            updated_df[column]=label_encoder.fit_transform(updated_df[column])
    X=updated_df.drop("gender",axis=1)     
    Y=updated_df["gender"]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    knn_classifier=KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train,Y_train)
    
    Prediction=knn_classifier.predict(X_test)
    conf=confusion_matrix(Y_test,Prediction)
    recall=conf[0][0]/(conf[0][0]+conf[1][0])
    precision=conf[0][0]/(conf[0][0]+conf[0][1])
    f1=(2*(precision*recall))/(precision+recall)
    
    return (conf,recall,precision,f1)

#Mean absolute error 
def MAE(y_test,y_pred):
    N=len(y_test)
    diff_sum=0
    for i in range(N):
        diff_sum+=y_test[i]-y_pred[i]
    return diff_sum/N

#Mean Squared error
def MSE(y_test,y_pred):
    N=len(y_test)
    diff_sum_sq=0
    for i in range(N):
        diff_sum_sq+=(y_test[i]-y_pred[i])**2
    return diff_sum_sq/N

#Root mean squared error
def RMSE(y_test,y_pred):
    N=len(y_test)
    diff_sum_sq=0
    for i in range(N):
        diff_sum_sq+=(y_test[i]-y_pred[i])**2
    return (diff_sum_sq/N)**0.5

#R squared error
def R_sqr(y_test,y_pred):
    y_bar=np.mean(y_test)
    diff_sum_sq=0
    diff_mean_sq=0
    for i in range(len(y_test)):
          diff_sum_sq+=(y_test[i]-y_pred[i])**2
          diff_mean_sq+=(y_test[i]-y_bar)**2
          
    r2=1-(diff_sum_sq)/(diff_mean_sq)
    return r2

#Find Mae, MSE, RMSE and R2 errors
def q2():
    df=pd.read_excel('Lab Session1 Data (1).xlsx')
    df=df.iloc[:,1:6]
    X=df.drop('label')
    y=df['label']
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,Y_train)
    Prediction=knn.predict(X_test)
    mse=MSE(list(Y_test),list(Prediction))
    mae=MAE(list(Y_test),list(Prediction))
    rmse=RMSE(list(Y_test),list(Prediction))
    r2=R_sqr(list(Y_test),list(Prediction))
    
    return (mse,mae,rmse,r2)

#Generate training dataset and plots it
def question_A3_trainingdata():
    data_points=np.random.randint(1,11,size=(20,2))
    X=data_points[:,0]
    Y=data_points[:,1]
    classXY=[]
    for i in range(len(X)):
        if ((X[i]+Y[i])%2)==0:#If the sum of the two features is odd ->class 1 if even -> class 0
            classXY.append((X[i],Y[i],0))
        else:
            classXY.append((X[i],Y[i],1))
       
    for i in classXY:
        if i[-1]==0:
            c='b'
            plt.scatter(i[0],i[1],color=c)
        else:
            c='r'
            plt.scatter(i[0],i[1],color=c)
    plt.show()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Training data")
    return classXY

#Generates Testing dataset
def question_A4_testset(k=3):
   train_data=question_A3_trainingdata()
   X_train=[]
   Y_train=[]
   for i in train_data:
       X_train.append((i[0],i[1]))
       Y_train.append(i[-1])
   
   test_data=[]
   x_values=np.linspace(0,10,10000)
   y_values=np.linspace(0,10,10000)
   for i in range(len(x_values)):
       test_data.append((x_values[i],y_values[i]))
   knn_classifier=KNeighborsClassifier(n_neighbors=k)
   knn_classifier.fit(X_train,Y_train)
   y_pred=knn_classifier.predict(test_data)

   for i in range(len(y_pred)):
        if y_pred[i]==0:
            c='b'
            plt.scatter(test_data[i][0],test_data[i][1],color=c)
        else:
            c='r'
            plt.scatter(test_data[i][0],test_data[i][1],color=c)
   plt.show()
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.title("Testing data")
   
#Applies knn classifier for various values of k given by user.
def question_A5_testK(max_value_of_k):
    for i in range(3,max_value_of_k+1):
        question_A4_testset(i)

#Applies q3 to 6 for 2 features and classes of fashion dataset.
def question_A6(k):
    df=pd.read_excel('Styles.xlsx')
    cols=['season','usage','gender']
    new_df=df[cols]
    class_men=new_df[new_df['gender']=='Men']
    class_women=new_df[new_df['gender']=='Women']
    updated_df=class_men._append(class_women,ignore_index=True)
    label_encoder=LabelEncoder()
    for column in updated_df.columns:
        if updated_df[column].dtype=='object':
            updated_df[column]=label_encoder.fit_transform(updated_df[column])
    X=updated_df.drop("gender",axis=1)     
    Y=updated_df["gender"]
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    for i in range(len(X_train)):
        if Y_train.iloc[i]==0:
            plt.scatter(X_train.iloc[i,0],X_train.iloc[i,1],color='b')
        else:
            plt.scatter(X_train.iloc[i,0],X_train.iloc[i,1],c='r')

    plt.show()
    plt.xlabel("X-train")
    plt.ylabel("Y-train")
    plt.title("Training data")
    for i in range(k):
        knn_classifier=KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train,Y_train)
        Prediction=knn_classifier.predict(X_test)
        for i in range(len(Prediction)):
            if Prediction[i]==0:
                plt.scatter(X_test.iloc[i,0],X_test.iloc[i,1],color='b')
            else:
                 plt.scatter(X_test.iloc[i,0],X_test.iloc[i,1],c='r')
        plt.show()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Testing data")
 
#Uses RandomSearchCV() to find the optimal value of k.
def question_7():
    df=pd.read_excel('Styles.xlsx')
    cols=['season','usage','gender']
    new_df=df[cols]
    class_men=new_df[new_df['gender']=='Men']
    class_women=new_df[new_df['gender']=='Women']
    updated_df=class_men._append(class_women,ignore_index=True)
    label_encoder=LabelEncoder()
    for column in updated_df.columns:
        if updated_df[column].dtype=='object':
            updated_df[column]=label_encoder.fit_transform(updated_df[column])
    X=updated_df.drop("gender",axis=1)     
    Y=updated_df["gender"]
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    model=KNeighborsClassifier()
    param_dist={'n_neighbors':range(5)}
    random_search=RandomizedSearchCV(estimator=model,param_distributions=param_dist,n_iter=100,cv=5)
    random_search.fit(X_train,Y_train)
    best_params=random_search.best_params_
    best_model=random_search.best_estimator_
    return(best_model)
def main():
    op=int(input('''Enter the option: 
                    1. Confusion matrix problem
                    2. MSE,RMSE, MAPE,R2 scores
                    3. Generate 20 data points - training set data
                    4. Testing set data
                    5. Different values of K
                    6. A3-A5 for fashion dataset
                    7. RandomSearchCV() and GridSearchCV()'''))
    if op==1:
        (conf,recall,precision,f1)=question_A1()
        print("Confusion matrix:")
        print(conf)
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"F1-score: {f1}")
        
    elif op==2:
        (mse,mae,rmse,r2)=q2()
        print(f"Mean squared error: {mse}")
        print(f"Mean Absolute error: {mae}")
        print(f"Root mean squared error: {rmse}")
        print(f"R squared error: {r2}")
        
    elif op==3:
        classes=question_A3_trainingdata()
        print("Class distribution: ")
        print(classes)
        
    elif op==4: 
        k=int(input("Enter the value of k: "))
        question_A4_testset(k)
        
    elif op==5:
        max_k=int(input("Enter the maximum value of k: "))
        question_A4_testset(max_k)
        
    elif op==6:
        k=int(input("Enter the maximum value of k: "))
        question_A6(k)
        
    elif op==7:
        k=question_7()
        print("The best model is ")
        print(k)
        
    else:
        print("Invalid option!")
        
main()