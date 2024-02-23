#R. Sanjana - BL.EN.U4AIE22149
import numpy as np
import pandas as pd
import sklearn
import scipy
from numpy import linalg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Question -1 
def Interclass_and_Intraclassspread(df,class1,class2):
    df_class1=df[df['Species'].isin([class1])]
    df_class2=df[df['Species'].isin([class2])]
    cols_to_calculate1=df_class1.iloc[:,1:5]
    cols_to_calculate2=df_class2.iloc[:,1:5]
    matrix1=cols_to_calculate1.values
    matrix2=cols_to_calculate2.values
    means1=[] #Centroid of Class 1
    means2=[] #Centroid of Class 2
    std1=[]
    std2=[]
    for i in range(len(matrix1[0])):
        means1.append((np.mean(matrix1[:,i],axis=0)))
        std1.append(np.std(matrix1[:,i],axis=0))
        means2.append((np.mean(matrix2[:,i],axis=0)))
        std2.append(np.std(matrix2[:,i],axis=0))
    
    distance_means=np.linalg.norm(means2,means1)
    return (distance_means)

#Question-2
def density_pattern(df,feature,bins_input):
    hist, bin_edges=np.histogram(df[feature],bins=bins_input)
    sns.histplot(data=df,x=feature,bins=bins_input)
    plt.title(feature)
    plt.show()
    mean_hist=np.mean(hist)
    var_hist=np.var(hist)
    return((mean_hist,var_hist))

#Question-3
def Minkwoski_distance(f1,f2):
    values=[]
    r_vals=range(1,11)
    for r in r_vals:
        sum_for_r=0
        for i in range(len(f1)):
            diff=float(f1[i])-float(f2[i])
            diff_pow=abs(diff)**r
            sum_for_r+=diff_pow
        values.append(sum_for_r**(1/r))
    plt.scatter(r_vals,values)  
    plt.xlabel("r")
    plt.ylabel("Minkwoski Distance")
    plt.table("Minkwoski distances of feature vectors.")
    plt.show()
    
#Question-4
def split_dataset(df,class1,class2,test_size_input):
    df_class=df[df["Species"].isin((class1,class2))]
    features=df_class.drop(columns=['Species'])
    Species=df_class['Species']
    print(features.shape)
    print(Species.shape)
    features_train,features_test,Species_train,Species_test=train_test_split(features,Species,test_size=test_size_input)
    
    return ((features_train,features_test,Species_train,Species_test))

#Question-5
def Train_Knn(df,class1,class2,test_size_input,k):
    (X_train,X_test,Y_train,Y_test)=split_dataset(df,class1,class2,test_size_input)
    scaler=StandardScaler()
    X_train_scale=scaler.fit_transform(X_train)
    X_train_scale=scaler.transform(X_test)
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train_scale,Y_train)
    
    return (neigh,X_test,Y_test)

#Question-7
def predict_model(df,class1,class2,test_size_input,k):
    (model,X_test,Y_test)=Train_Knn(df,class1,class2,test_size_input,k)
    scaler=StandardScaler()
    X_test_scale=scaler.fit_transform(X_test)
    X_test_scale=scaler.transform(X_test)
    prediction=model.predict(X_test_scale)
    return (model,prediction)

#Question-6
def Test_accuracy(df,class1,class2,test_size_input,k):
    (model,prection)=predict_model(df,class1,class2,test_size_input,k)
    accuracy=model.score(X_test_scale,Y_test)
    return accuracy
  
#Question-8  
def k_1or3(df,class1,class2,test_size_input,k1=1,k2=3):
    (model1,prediction1)=Test_accuracy(df,class1,class2,test-size_input,k1) #Model with k=1
    (model3,prediction3)=Test_accuracy(df,class1,class2,test-size_input,k3) #Model with k=1
    accuracy=[]
    r=range(1,12)
    for k in r:
        (modelk,predictionk)=Test_accuracy(df,class1,class2,test-size_input,k) #Model with k
        accuracy.append(predictionk)
    accuracy=[]
    plt.scatter(r,accuracy)
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("k v/s Accuracy")
    plt.show
    
    return((prediction1,prediction3))

#Question-9
def confusion_mat(df,class1,class2,test_size_input,k):
    df_class=df[df["Species"].isin((class1,class2))]
    features=df_class.drop(columns=['Species'])
    Species=df_class['Species']
    features_train,features_test,Species_train,Species_test=train_test_split(features,Species,test_size=test_size_input)
    scaler=StandardScaler()
    X_train_scale=scaler.fit_transform(features_train)
    X_train_scale=scaler.transform(features_test)
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train_scale,Species_train)
    X_test_scale=scaler.fit_transform(X_test)
    X_test_scale=scaler.transform(X_test)
    prediction=model.predict(X_test_scale)
    conf_matrix=confusion_matrix(Species_train,prediction)
    
    return conf_matrix

def main():
    df=pd.read_excel("Dataset.xlsx") #Opens and reads the dataset.
    clean_df=df.dropna(axis=0,inplace=False) #Removes rows with NaN values 
    op=int(input('''Enter the option: 
                    1. Interclass and Intraclass spread.
                    2. Histogram of feature.
                    3. Minkwoski distance
                    4. Split dataset
                    5. Train dataset
                    6. Accuracy
                    9. Confusion matrix
                 '''))
    if op==1:
        c1=input("Enter class 1: ")
        c2=input("Enter class 2:")
        interclass_spread=Interclass_and_Intraclassspread(clean_df,c1,c2)
        print(f"The interclass spread is {interclass_spread}.")
    elif op==2:
        feature=input("Enter feature name: ")
        bins=int(input("Enter the number of bins: "))
        (mean,var)=density_pattern(clean_df,feature,bins)
        print(f"Mean of the density pattern is {mean}.")
        print(f"Variance of the density pattern is {var}.")
    elif op==3:
        print("Order:SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm ")
        f_vector1=input("Enter the feature vector one separated by commas: ")
        feature_1=[float(x) for x in f_vector1.split(',')]
        f_vector2=input("Enter the feature vector two separated by commas: ")
        feature_2=[float(x) for x in f_vector2.split(',')]
        Minkwoski_distance(feature_1,feature_2)
        
    elif op==4:
        class1=input("Enter class 1: ")
        class2=input("Enter class 2: ")
        test_size=float(input("Enter the size of test set within (0,0.9): "))
        (X_train,X_test,Y_train,Y_test)=split_dataset(clean_df,class1,class2,test_size)
        
    elif op==5:
        c1=input("Enter class 1: ")
        c2=input("Enter class 2: ")
        test_size=float(input("Enter the size of test set: "))
        k=int(input("Enter the size of k: "))
        (a,b,c)=Train_Knn(clean_df,c1,c2,test_size,k)
        print(a)
        
    elif op==6:
        c1=input("Enter class 1: ")
        c2=input("Enter class 2: ")
        test_size=float(input("Enter the size of test set: "))
        k=int(input("Enter the size of k: "))
        accuracy=Test_accuracy(clean_df,c1,c2,test_size,k)
        print(f"Accuracy of model is {accuracy}.")
        
    elif op==7:
        c1=input("Enter class 1: ")
        c2=input("Enter class 2: ")
        test_size=float(input("Enter the size of test set: "))
        k=int(input("Enter the size of k: "))
        print("Prediction: ")
        print(predict_model(clean_df,c1,c2,test_size,k))
    
    elif op==8:
        c1=input("Enter class 1: ")
        c2=input("Enter class 2: ")
        test_size=float(input("Enter the size of test set: "))
        (prediction1,prediction3)=k_1or3(clean_df,c1,c2,test_size,1,3)
        print(f"Accuracy for k=1 is {prediction1}.")
        print(f"Accuracy for k=3 is {prediction3}")
        
    elif op==9:
        c1=input("Enter class 1: ")
        c2=input("Enter class 2: ")
        test_size=float(input("Enter the size of test set: "))
        k=int(input("Enter the size of k: "))
        print("Confusion matrix: ")
        print(confusion_mat(clean_df,c1,c2,test_size,k))
    
    else:
        print("Invalid operation!")
main()