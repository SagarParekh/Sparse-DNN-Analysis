#!/usr/bin/env python3
from ctypes import *

mkl = cdll.LoadLibrary("/home/sagar/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin/libmkl_rt.so")
dgemm = mkl.cblas_dgemm

def print_mat(mat, m, n):
  for i in range(0,m):
    print (" ")
    for j in range(0,n):
      print (mat[i*n+j]),
    print

Order = 101  # 101 for row-major, 102 for column major data structures
TransA = 111 # 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
TransB = 111
m = 2
n = 4
k = 3
lda = k
ldb = n
ldc = n
alpha = 1.0
beta = 0

amat = c_double * 6
bmat = c_double * 12
cmat = c_double * 8
a = amat(1,2,3, 4,5,6)
b = bmat(0,1,0,1, 1,0,0,1, 1,0,1,0)
c = cmat(5,1,3,3, 11,4,6,9)

print ("\nMatrix A =")
print_mat(a,2,3)
print ("\nMatrix B =")
print_mat(b,3,4)
print ("\nMatrix C =")
print_mat(c,2,4)

print ("\nCompute", alpha, "* A * B + ", beta, "* C")

dgemm( c_int(Order), c_int(TransA), c_int(TransB), c_int(m), c_int(n), c_int(k), c_double(alpha), byref(a), c_int(lda), byref(b), c_int(ldb), c_double(beta), byref(c), c_int(ldc))

print_mat(c,2,4)
print
