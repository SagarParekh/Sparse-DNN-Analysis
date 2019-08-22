all:
	icc -mkl -O3 matMul_dgemm.cpp
	./a.out

clean:
	-@rm -rvf a.out
	-@rm -rvf input_a.txt
	-@rm -rvf input_b.txt
	-@rm -rvf output_c.txt
