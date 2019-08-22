#!/usr/bin/env python3
import sys
import numpy as np
from ctypes import *

# Load the share library
mkl = cdll.LoadLibrary("/home/sagar/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin/libmkl_rt.so")
# For Intel MKL prior to version 10.3 us the created .so as below
# mkl = dll.LoadLibrary("./libmkl4py.so")
cblas_dgemm = mkl.cblas_dgemm


def print_mat(mat, m, n):
  for i in range(0,m):
    print (" "),
    for j in range(0,n):
      print (mat[i*n+j]),
    print

# Initialize scalar data
Order = 101  # 101 for row-major, 102 for column major data structures
TransA = 111 # 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
TransB = 111

f = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_dgemm.txt').read()
f = f.split(' ')
m = int(f[0])
print("M is:",m)
p = int(f[1])
n = int(f[2])
#m = 2
#n = 4
#k = 3
lda = n
ldb = p
ldc = p

alpha = 1.0
beta = 0 # it was -1 originally

# Create contiguous space for the double precision array
# amat = c_double * 6
# bmat = c_double * 12
# cmat = c_double * 8

# Initialize the data arrays
# a = amat(1,2,3, 4,5,6)
# # print(a)
# b = bmat(0,1,0,1, 1,0,0,1, 1,0,1,0)
# c = cmat(5,1,3,3, 11,4,6,9)
#
# print ("Matrix A =")
# # print_mat(a,2,3)
# print ("Matrix B =")
# # print_mat(b,3,4)
# print ("Matrix C =")
# # print_mat(c,2,4)

f = open('/home/sagar/Sparse-DNN-Analysis/python_wrapper/input_a_for_wrapper.txt')
A=f.read().split(',')
for line in
# A = [A[i * p:(i + 1) * p] for i in range((len(A) + p - 1) // p )]
# A=np.array(A,dtype=int)
print(A)
# print("hereee",len(A))

# print ("Matrix A =")
# print_mat(a,m,p)

f = open('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/dgemm/input_b.txt').read()
B=f.split()
# B = [B[j * n:(j + 1) * n] for j in range((len(B) + n - 1) // n )]
# B=np.array(B,dtype=int)
# np.savetxt('/home/sagar/Sparse-DNN-Analysis/matrix_multiplication/test/python_output/large_input_B.txt',B,fmt='%.2f')
#
print ("Compute", alpha, "* A * B + ", beta, "* C")
#
# # Call Intel MKL by casting scalar parameters and passing arrays by reference
cblas_dgemm( c_int(Order), c_int(TransA), c_int(TransB),
             c_int(m), c_int(p), c_int(n), c_double(alpha), byref(A), c_int(lda),
             byref(B), c_int(ldb), c_double(beta), byref(c), c_int(ldc))
#
print(c)
# print_mat(c)
print
