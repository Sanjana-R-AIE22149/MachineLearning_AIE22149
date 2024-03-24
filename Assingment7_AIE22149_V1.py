from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def percept():
    X=[[0,0],[0,1],[1,0],[1,1]]
    y=[0,1,1,0]
    
    perceptron=Perceptron()
    perceptron.fit(X,y)
    predictions=perceptron.predict(X)
    print(predictions)
    print(perceptron.score(X,y))
    para_grid={
        'alpha':[0.0001,0.005,0.01,0.1],
    }
    randsearch=RandomizedSearchCV(estimator=perceptron, param_distributions=para_grid, n_iter=4, cv=2, random_state=42)
    randsearch.fit(X,y)
    print(randsearch.best_params_)
    # Access the best model
    best_model = randsearch.best_estimator_

    accuracy = best_model.score(X,y)
    print("Accuracy:", accuracy)

def mlp():
    X=[[0,0],[0,1],[1,0],[1,1]]
    y=[0,1,1,0]
    mlp=MLPClassifier()
    mlp.fit(X,y)
    predictions=mlp.predict(X)
    print(predictions)
    print(mlp.score(X,y))
    para_grid = {
        'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
    }

    randsearch=RandomizedSearchCV(estimator=mlp, param_distributions=para_grid, n_iter=4, cv=2, random_state=42)
    randsearch.fit(X,y)
    print(randsearch.best_params_)
    best_model=randsearch.best_estimator_

    accuracy = best_model.score(X,y)
    print("Accuracy:", accuracy)

def multiclass_MLP():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']    
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    mlp=MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.01, max_iter=1000)
    mlp.fit(X_train,Y_train)
    predict_mlp=mlp.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_mlp)
    precision=precision_score(Y_test,predict_mlp)
    recall=recall_score(Y_test,predict_mlp)
    f1=f1_score(Y_test,predict_mlp)
    
    return(accuracy,precision, recall,f1)
def multiclass_perceptron():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']    
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    perceptron=Perceptron()
    perceptron.fit(X_train,Y_train)
    predict_perceptron=perceptron.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_perceptron)
    precision=precision_score(Y_test,predict_perceptron)
    recall=recall_score(Y_test,predict_perceptron)
    f1=f1_score(Y_test,predict_perceptron)
    
    return(accuracy,precision, recall,f1)
    
def multiclass_SVM():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    svc=SVC()
    svc.fit(X_train,Y_train)
    predict_SVC=svc.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_SVC)
    precision=precision_score(Y_test,predict_SVC)
    recall=recall_score(Y_test,predict_SVC)
    f1=f1_score(Y_test,predict_SVC)
    
    return(accuracy,precision, recall,f1)   

def multiclass_DecisionTree():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    dt=DecisionTreeClassifier()
    dt.fit(X_train,Y_train)
    predict_dt=dt.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_dt)
    precision=precision_score(Y_test,predict_dt)
    recall=recall_score(Y_test,predict_dt)
    f1=f1_score(Y_test,predict_dt)
    
    return(accuracy,precision, recall,f1)   

def multiclass_RandomForest():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    rf=RandomForestClassifier()
    rf.fit(X_train,Y_train)
    predict_rf=rf.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_rf)
    precision=precision_score(Y_test,predict_rf)
    recall=recall_score(Y_test,predict_rf)
    f1=f1_score(Y_test,predict_rf)
    
    return(accuracy,precision, recall,f1)     

def multiclass_CatBoost():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)   
    cb=CatBoostClassifier()
    cb.fit(X_train,Y_train)
    predict_cb=cb.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_cb)
    precision=precision_score(Y_test,predict_cb)
    recall=recall_score(Y_test,predict_cb)
    f1=f1_score(Y_test,predict_cb)
    
    return(accuracy,precision, recall,f1) 

def multiclass_NaiveBayes():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2) 
    nb=GaussianNB()
    nb.fit(X_train,Y_train)
    predict_nb=nb.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_nb)
    precision=precision_score(Y_test,predict_nb)
    recall=recall_score(Y_test,predict_nb)
    f1=f1_score(Y_test,predict_nb)
    
    return(accuracy,precision, recall,f1)   

def multiclass_AdaBoost():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2) 
    ab=AdaBoostClassifier()
    ab.fit(X_train,Y_train)
    predict_ab=ab.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_ab)
    precision=precision_score(Y_test,predict_ab)
    recall=recall_score(Y_test,predict_ab)
    f1=f1_score(Y_test,predict_ab)
    
    return(accuracy,precision, recall,f1)   
    
def multiclass_XGBoost():
    df=pd.read_excel('Styles.xlsx')

    label_encoder=LabelEncoder()

    for column in df.columns:
        if df[column].dtype=='object':
            df[column]=label_encoder.fit_transform(df[column])
    X=df.drop('gender',axis=1)
    y=df['gender']     
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2) 
    xgb=XGBClassifier()
    xgb.fit(X_train,Y_train)
    predict_xgb=xgb.predict(X_test)
    accuracy=accuracy_score(Y_test,predict_xgb)
    precision=precision_score(Y_test,predict_xgb)
    recall=recall_score(Y_test,predict_xgb)
    f1=f1_score(Y_test,predict_xgb)
    
    return(accuracy,precision, recall,f1)   
    
def main():
    op=int(input('''
                 1. Perceptron optimization
                 2. MLP optimization
                 3. Multiclass performance
                 '''))
    if op==1:
        percept()
    if op==2:
        mlp()
        
    if op==3:
       print('''
Classifier Name     Accuracy    Precision   Recall     F1 score
''') 
       print("Perceptron",multiclass_perceptron())
       print("MLP", multiclass_MLP())
       print("SVM",multiclass_SVM())
       print("Decision Tree",multiclass_DecisionTree())
       print("Random Forest",multiclass_RandomForest())
       print("Naive Bayes",multiclass_NaiveBayes())
       print("AdaBoost: ",multiclass_AdaBoost())
       print("XGBoost",multiclass_XGBoost())
       print("Catboost",multiclass_CatBoost())
main()