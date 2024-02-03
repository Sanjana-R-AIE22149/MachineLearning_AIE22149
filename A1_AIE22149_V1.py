#Question-1 : Given a list of integers, find all pairs of integers such that their sum is 10.


def find_pairs_of_ten(input_list):
    pairs_of_sum10=[]
    #i and j are traversal variables.
    #i traverses the list.
    for i in range(len(input_list)):
     curr_val=input_list[i]
    
     for j in range(len(input_list)):  #j traverses the list to compare elements with the current value (curr_value) to find sum as 10.
         if input_list[j]+curr_val==10:
              pairs_of_sum10.append((curr_val,input_list[j])) #Append the pairs of sum=10 to a list
    return pairs_of_sum10 #Return the list with pairs of integers of sum 10.

#Question-2: Given a list of integers, find the range which is the difference betweent the maximum and minimum value in the list.

#To find the minimum value in a list.
def minimum(input_list):
    min=10000 #A arbitrary large value to compare and find minimum.
    for i in range(len(input_list)):
        if input_list[i]<min:
            min=input_list[i]
    return min #Return the minimum value from the given list.
 
#To find the maximum value in a list.
def maximum(input_list):
    max=-100000 #A arbitraty small value to compare and find maximum.
    for i in range(len(input_list)):
        if input_list[i]>max:
            max=input_list[i]
    return max #Return the maximum value from the given list.

#Finds the range of the given list.
def find_list_range(input_list):
    #If the length of the list is less than 3, range cannot be determined. 
    if len(input_list)<3:
        return None
    #Find the minimum and maximum values of the list.
    min_value=minimum(input_list) 
    max_value=maximum(input_list)
    range=max_value-min_value
    return range #Return the range.

#Question-3: Given a square matrix A and an integer m, find A^m.
#To create a zero matrix with size of the matrix passed as the argument.
def create_matrix(matrix):
    matrix_2=[]
    for i in range(len(matrix)): #Traverses the rows
        row=[]
        for j in range(len(matrix[0])): #Traverses the columns
            row.append(0)
        matrix_2.append(row)
    return matrix_2 

#To multiply two matrices.
def multiply_matrices(matrix1,matrix2):
    result_matrix=create_matrix(matrix1) #A zero matrix of the size of matrix1 as both the matrices are of same size in the qn.
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result_matrix[i][j]=result_matrix[i][j]+matrix1[i][k]*matrix2[k][j]
    return result_matrix

#Multiplies the matrix with itself, power times to get A^m.
def find_matrix_power(matrix,power_value):
    original_matrix=matrix.copy()
    for i in range(power_value-1):
        matrix=multiply_matrices(matrix,original_matrix)
    return matrix

#Question-4: Given a string, find the most occuring letter and its occurence count.
def find_most_occuring_letter(input_string):
    char_list=[]
    #Add each unique letter in the string to a list
    for i in input_string:
        if i not in char_list:
            char_list.append(i)
            
    #Create a dictionary of unique letters in the string and their count. Count is taken to be 0 initially.
    char_count={char:0 for char in char_list}
    
    #Find the count of each letter in the string.
    for current_letter in input_string:
        count=0
        for compare_letter in input_string:
            if current_letter==compare_letter:
                count=count+1
        char_count[current_letter]=count
    #To find the highest frequency of letter.    
    max_value=max(char_count.values())
    most_occuring_char=None
    for key, value in char_count.items():
        if value==max_value:
            most_occuring_char=key #To find the letter with highest frequency.
            
    return (most_occuring_char, max_value)

#Main function
def main():
    #Menu to select the question.
    print("Which function?")
    op=int(input('''
                 1. Find pairs of numbers in a list such that sum is 10.
                 2. Find the range in a list of numbers.
                 3. Find A power m for a square matrix A and integer m.
                 4. Find the highest occuring character in a string and its count.
                 '''))
    
    if op==1:
        number_of_numbers=int(input("Enter the number of numbers in list: "))
        input_list=[]
        for i in range(number_of_numbers):
            number=int(input("Enter the number: "))
            input_list.append(number)
            
        pairs_of_sum10=find_pairs_of_ten(input_list)
        print(f"The pairs of numbers in list such that their sum is 10 are {pairs_of_sum10}.")
            
    elif op==2:
        input_list=input("Enter the list items separated by comma:")
        input_list_for_range=[int(x) for x in input_list.split(",")]
        range_of_list=find_list_range(input_list_for_range)
        if range_of_list==None:
            print("Range determination is not possible.")
        else:
            print(f"The range is {range_of_list}.")
    
    elif op==3:
        #Takes number of rows, columns, the power and the matrix as input and prints the output.
        row_val=int(input("Enter the number of rows: "))
        col_val=int(input("Enter the number of columns: "))
        matrix=[]
        for i in range(0,row_val):
            row=[]
            for j in range(0,col_val):
                value=int(input(f"Enter the value in index {i}{j}:"))
                row.append(value)
            matrix.append(row)
        power_value=int(input("Enter the power to raise the matrix to: "))
        result_matrix=find_matrix_power(matrix,power_value)
        print(f"Matrix A to the power {power_value} is")
        print(result_matrix)
        
    elif op==4:
        input_string=input("Enter a string: ")
        maximal_char,count=find_most_occuring_letter(input_string)
        print(f"The maximally occuring character is {maximal_char} and its occurence count is {count}.")
        
    else: #If any option other than 1-4 is picked.
        print("Pick a valid option.")
            
if __name__==main():
    main()
    