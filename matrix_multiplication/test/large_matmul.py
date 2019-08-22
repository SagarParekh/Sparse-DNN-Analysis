#!/usr/bin/env python3
import sys
import numpy as np

f = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_dgemm.txt').read()
f = f.split(' ')
p = int(f[1])
n = int(f[2])

f = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_a.txt').read()
A=f.split()
A = [A[i * p:(i + 1) * p] for i in range((len(A) + p - 1) // p )]
A=np.array(A,dtype=int)
np.savetxt('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/test/python_output/large_input_A.txt',A,fmt='%.2f')

f = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_b.txt').read()
B=f.split()
B = [B[j * n:(j + 1) * n] for j in range((len(B) + n - 1) // n )]
B=np.array(B,dtype=int)
np.savetxt('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/test/python_output/large_input_B.txt',B,fmt='%.2f')

C = A.dot(B)
np.savetxt('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/test/python_output/large_output_C.txt',C,fmt='%.2f')
print("Python file Executed.")

# def matrix(file):
# 	contents = open(file).read()
# 	return [item.split() for item in contents.split('\n')[:-1]]
