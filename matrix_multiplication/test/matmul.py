#!/usr/bin/env python3

# input two matrices of size n x m 
matrix1 = [[183, 176, 80, 191, 206],
 	   [192, 127, 112, 217, 183],
	   [110, 168, 182, 133, 226],
	   [187, 194, 56, 189, 142],
	   [58, 216, 199, 194, 176]] 
matrix2 = [[172, 148, 60, 171, 159],
	  [225, 226, 80, 49, 33],
	  [157, 241, 31, 13, 202],
	  [213, 249, 241, 12, 126],
	  [212, 198, 191, 139, 4]] 

res = [[0 for x in range(5)] for y in range(5)] 

# explicit for loops 
for i in range(len(matrix1)): 
	for j in range(len(matrix2[0])): 
		for k in range(len(matrix2)): 

			# resulted matrix 
			res[i][j] += matrix1[i][k] * matrix2[k][j]
			
print (res) 

