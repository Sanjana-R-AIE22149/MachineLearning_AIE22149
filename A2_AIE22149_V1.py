import math 
#Question-1: Find the Euclidean and Manhattan distance
def Euclidean_distance(input_vector1,input_vector2):
    sum_square_of_differences=0
    for i in range(0,len(input_vector1)):
        #Traverse the two vectors taking the corresponding elements
        x=float(input_vector1[i]) 
        y=float(input_vector2[i])
        #Find the square of difference between the corresponding elements and it to total sum.
        sum_square_of_differences+=(y-x)*(y-x)
        #Return the square root of the sum.
    return math.sqrt(sum_square_of_differences)

def Manhattan_distance(input_vector1, input_vector2):
    manhattan_dist=0
    for i in range(len(input_vector1)):
        coordinate_1=input_vector1[i]
        coordinate_2=input_vector2[i]
        #Find the difference between the corresponding elements and add to sum of all differences
        manhattan_dist+=(coordinate_2-coordinate_1)
        #Return the Manhattan distance.
    return manhattan_dist


#Question-2: K-nn classifier
def Knn(input_vector,k):
    #Open and read the contents of the csv file.
    f=open('Dataset.csv','r')
    data=f.read()
    lines=data.splitlines()
    distances={}
    dist=[]
    #Dictionary of count of each possible classes.
    classes_counts={'Iris-setosa':0,'Iris-versicolor':0,'Iris-virginica':0}
    #Traverse each line except the header line
    for line in lines[1:]:
        #Split the line at commas and exclude the label column (last column).
        vector_1=line.split(',')[:-1]
        #Find the euclidean distance between the current vector and input vector
        dist_curr_vector=Euclidean_distance(vector_1,input_vector)
        
        dist.append(dist_curr_vector)
        
    #Store the index number and corresponding euclidean distance in a dictionary.
    distances={index:dist[index] for index in range(len(dist))}
    top_k={}
    
    #Sort the distances vector using a lambda as the euclidean as the criteria.
    distances_sorted=dict(sorted(distances.items(), key=lambda item:item[1]))
    top_matches=[]
    #List of the list indices sorted.
    sorted_indices=list(distances_sorted.keys())
    for i in range(k):
        #Add the k-nearest neighbours to the list.
        top_matches.append((sorted_indices[i],distances_sorted[i]))
    
    class_labels=[lines[top_matches[index][0]+1].split(',')[-1] for index in range(k)]
    
    #Calculate the count of each class from the k nearest neighbours.
    for i in class_labels:
        if i in classes_counts.keys():
            classes_counts[i]+=1
    max_value=max(classes_counts.values()) 
    final_class=None
    #Find the class with maximum count and return the classification.
    for key, value in classes_counts.items():
        if value==max_value:
            final_class=key
            
    return(final_class)
#Question-3: Label Encoding
#Read the file and read it.
def Label_Encoding():
    f=open('Dataset.csv','r')
    data=f.read()
    lines=data.splitlines()
    classes=[]
    #Traverse through all the records of the csv file except header line and all unique class names.
    for i in lines[1:]:
        field=i.split(',')
        label=field[-1]
        if label not in classes:
            classes.append(label)
    #Assign a number(through iteration) for every unique class and return the encoding.      
    class_encoding={classes[j]:j for j in range(len(classes))}
    return(class_encoding)
#Question-4: One hot encolding 
def One_Hot_Encoding():
    #Open and read the csv file.
    f=open('Dataset.csv','r')
    data=f.read()
    lines=data.splitlines()
    classes=[]
    #Traverse throught the entries of the csv file and form a list of all unique classes
    for i in lines[1:]:
        field=i.split(',')
        label=field[-1]
        if label not in classes:
            classes.append(label)

    #Assign a number (through iteration) for every unique class and change it to binary.
    max_len=len(bin(len(classes)-1)) #Find the maximum length of the binary version of encodings. 
    encoding={classes[j]:str(bin(j).zfill(max_len)).replace('b','') for j in range(len(classes))} #Remove the 'b' which comes in bin format and fill with 0's so that all have same length.
    return(encoding)
    
    #All the necessary user inputs are taken in the main function, and the corresponding functions are called and result is printed in the main().
def main():
    option=int(input('''
                     Enter the function option:
                     1. Euclidean Distance of 2 vectors.
                     2. Manhattan Distance of 2 vectors.
                     3. K-NN classifier.
                     4. Label Encoding of the Iris Dataset.
                     5. One Hot Encoding of the Iris Dataset.
                     '''))
    if option==1:
        vector_1=input("Enter vector 1, elements separated by semicolons: ")
        input_vector1=[int(x) for x in vector_1.split(';')]
        vector_2=input("Enter vector 2, elements separated by semicolons: ")
        input_vector2=[int(x) for x in vector_2.split(';')]
        Euc_dist=Euclidean_distance(input_vector1,input_vector2)
        print(f"Euclidean distance of the 2 vectors is {Euc_dist}.")
    elif option==2:
        vector1=input("Enter a vector, separated by commas: ")
        input_vec1=[int(i) for i in vector1.split(',')]
        vector2=input("Enter a vector, separated by commas: ")
        input_vec2=[int(i) for i in vector2.split(',')]
        Manhat_dist=Manhattan_distance(input_vec1,input_vec2)
        print(f"Manhattan distance of the 2 vectors is {Manhat_dist}.")
    elif option==3:
        input_vector=[]
        data_list=input("Enter the details of the flower to identify in the following order:Id SepalLength (in Cm),SepalWidth (in Cm), PetalLength (in Cm),PetalWidth (in Cm)")
        input_vector=[float(j) for j in data_list.split()]
        k=int(input("Enter the number of nearest neighbours to consider: "))
        classification=Knn(input_vector,k)
        print(f"The given flower is a {classification}.")
    elif option==4:
        print(f"The label encoding of the Iris Dataset: {Label_Encoding()}")    
        
    elif option==5:
        print(f"The One Hot Encoding for the Iris Dataset is: {One_Hot_Encoding()}")
        
    else:
        print("Invalid option!")

main()