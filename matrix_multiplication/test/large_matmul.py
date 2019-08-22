#!/usr/bin/env python3

import numpy as np

# A = np.random.randint(low=1,high=210001000,size=(1000,1000))
# np.savetxt('python_output/large_input_A.txt',A,fmt='%.2f')
# B = np.random.randint(low=1,high=210001000,size=(1000,1000))
# np.savetxt('python_output/large_input_B.txt',B,fmt='%.2f')

f = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_a.txt').read()
A=f.split()
A = [A[i * 1000:(i + 1) * 1000] for i in range((len(A) + 1000 - 1) // 1000 )]
A=np.array(A,dtype=int)
np.savetxt('python_output/large_input_A.txt',A,fmt='%.2f')

f1 = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_b.txt').read()
B=f1.split()
B = [B[j * 1000:(j + 1) * 1000] for j in range((len(B) + 1000 - 1) // 1000 )]
B=np.array(B,dtype=int)
np.savetxt('python_output/large_input_B.txt',B,fmt='%.2f')

C = A.dot(B)
np.savetxt('python_output/large_output_C.txt',C,fmt='%.2f')


# def matrix(file):
# 	contents = open(file).read()
# 	return [item.split() for item in contents.split('\n')[:-1]]
