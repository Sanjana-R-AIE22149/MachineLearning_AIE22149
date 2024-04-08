import numpy as np
import pandas as pd

# def entropy():
#     df=pd.read_excel('Styles.xlsx')

# def root_node():
#     df=pd.read_excel('Styles.xlsx')
#     print(df.head)
#     target=df['gender']
#     features=df.drop('gender',axis=1)
#     print(features)
# root_node()

def binning(type,num_bins,data):
    if type=="equal_width":
        minval=min(data)
        maxval=max(data)
        width=(maxval-minval)/num_bins
        
        bins=[minval+i*width for i in range(num_bins)]
        arr=[]
        for i in range(0,num_bins-1):
            temp=[]
            for j in data:
                if bins[i]<=j<bins[i+1]:
                    temp=temp+[j]
            arr+=[temp]
        return (arr)
    elif type=="frequency":
        hist,bin_edges=np.histogram(data,bins=num_bins)
        bindata=np.digitize(data,bin_edges[:,-1])
        return bindata
    
def main():
    op=int(input("Enter the option: "))
    if op==2:
        bintype=input("Enter 1 for equal_width and 2 for frequency binning.")
        data=input("Enter data separated by spaces:")
        data=[int(x) for x in data.split()]
        num_bins=int(input("Enter the number of bins: "))
        if bintype=="1":
            print(binning("equal_width",num_bins,data))
        elif bintype=="2":
            print(binning("frequency",num_bins,data))
        else:
            print(binning("equal_width",num_bins,data)) #Default
            
main()