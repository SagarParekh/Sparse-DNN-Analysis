//g++ matMul.cpp -O3 -o matMul
// #include <boost/python.hpp>
// #include <Python.h>

#include <cstddef>
#include <chrono>
#include <ctime>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "mkl.h"

using namespace std;

double *A, *B, *C, *temp, *A2, *B2;
int m, n, p, i, j, r, mm;
double alpha, beta, densityA, densityB, sparsityA, sparsityB;
bool check;

int read_input(){
  double input[10];
  int x;
  fstream textfile;
  textfile.open("input_dgemm.txt");
  //Order of Inputs: m p n alpha beta densityA densityB
  while(! textfile.eof()){
    textfile >> input[x];
    if(textfile.eof()){
      break;
    }
    // cout << " " << input[x];
    x++;
  }
  textfile.close();
  m = input[0], p = input[1], n = input[2];
  //cout <<" Initializing data for matrix multiplication C=A*B for matrix "
  //        " A("<<m<<"x"<<p<<") and matrix B("<<p<<"x"<<n<<")\n"<<endl;
  alpha = input[3]; beta = input[4];
  densityA = input[5]; densityB = input[6];
  sparsityA = 1-densityA;
  sparsityB = 1-densityB;
  cout<<endl;
  cout<<" M is "<<m<<endl<<" P is "<<p<<endl<<" N is "<<n<<endl;
  cout<<" Alpha is "<<alpha<<endl<<" Beta is "<<beta<<endl;
  cout<<" Density of A is "<<densityA<<endl<<" Density of B is "<<densityB<<endl;
  //cout <<" Allocating memory for matrices aligned on 64-byte boundary for better performance\n"<<endl;
  A = new double[m*p*sizeof( double )];
  B = new double[p*n*sizeof( double )];
  C = new double[m*n*sizeof( double )];
  temp = new double[m*n*sizeof( double )];
  A2 = new double[m*p*sizeof( double )];
  // B2 = new double[p*sizeof( double )][n*sizeof( double )];

  if (A == NULL || B == NULL || C == NULL) {
    cout<<"\n ERROR: Can't allocate memory for matrices. Aborting...\n"<<endl;
    delete[] A;
    delete[] B;
    delete[] C;
    return 1;
  }
}

void init(){
    //cout <<" Intializing matrix data\n"<<endl;
    for (i = 0; i <(m*p); i++) {
        A[i] = 1+rand()%255;
    }
   for (i = 0; i < ceil(sparsityA*(m*p)); i++) {
//        A[rand()%(m*p)] = 0;
       do {
       r = rand()%(m*p);
       temp[i]=r;
       check=true;
   	for (int j=0;j<i;j++)
       	if (r == temp[j]) //if number is already used
       	{
           	check=false; //set check to false
           	break; //no need to check the other elements of value[]
       	}
   	} while (!check); //loop until new, unique number is found
       A[r] = 0;
   }
    for (i = 0; i <(p*n); i++) {
        B[i] = 1+rand()%255;
    }
   for (i = 0; i < ceil(sparsityB*(p*n)); i++) {
//        B[rand()%(p*n)] = 0;
       do {
       r = rand()%(p*n);
       temp[i]=r;
       check=true;
   	for (int j=0;j<i;j++)
       	if (r == temp[j]) //if number is already used
       	{
           	check=false; //set check to false
           	break; //no need to check the other elements of value[]
       	}
   	} while (!check); //loop until new, unique number is found
       B[r] = 0;
   }

    for (i = 0; i <(m*n); i++) {
        C[i] = 0.0;
    }
}

void matmul(){
    //cout << " Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n"<<endl;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, p, alpha, A, p, B, n, beta, C, n);
    cout<< "\n Computations completed.\n"<<endl;
}

void print_output(){
    cout <<"\n Top left corner of matrix A: \n";
    for (i=0; i<min(m,6); i++) {
        for (j=0; j<min(p,6); j++) {
            cout <<" "<< A[j+i*p] ;
        }
        cout<<"\n";
    }
    int counterA=0, counterB=0;
    for (i = 0; i <(m*p); i++) {
        if(A[i]!=0)
		counterA++;
    }
    cout<<"Number of NON-Zeros in A is: "<<counterA<<endl;
    cout <<"\n Top left corner of matrix B: \n";
    for (i=0; i<min(p,6); i++) {
        for (j=0; j<min(n,6); j++) {
            cout <<" "<< B[j+i*n] ;
        }
        cout<<"\n";
    }
    for (i = 0; i <(p*n); i++) {
        if(B[i]!=0)
		counterB++;
    }
    cout<<"Number of NON-Zeros in B is: "<<counterB<<endl;
    cout <<"\n Top left corner of matrix C: \n";
    for (i=0; i<min(m,6); i++) {
        for (j=0; j<min(n,6); j++) {
            cout <<" "<< C[j+i*n] ;
        }
        cout<<"\n";
    }
}

void write_output() {
	fstream myfile;
	myfile.open("input_a.txt",fstream::out);
	for (i=0; i<(m*p); i++) {
    myfile << A[i] << "\t";
    if((i+1)%p==0){
      myfile << "\n";
    }
  }
	myfile<<std::endl;
	myfile.close();

  myfile.open("input_b.txt",fstream::out);
	for (i=0; i<(p*n); i++) {
    myfile << B[i] << "\t";
    if((i+1)%n==0){
      myfile << "\n";
    }
  }
	myfile<<std::endl;
	myfile.close();

  myfile.open("output_c.txt",fstream::out);
	for (i=0; i<(m*n); i++) {
    myfile.precision(8);
    myfile << C[i] << "\t";
    if((i+1)%n==0){
      myfile << "\n";
    }
  }
	myfile<<std::endl;
	myfile.close();
}

void write_output_for_wrapper(){
  fstream myfile;
	myfile.open("/home/sagar/Sparse-DNN-Analysis/python_wrapper/input_a_for_wrapper.txt",fstream::out);
	for (i=0; i<(m*p); i++) {
    myfile << A[i];
    if(i==((m*p)-1)){
      break;
    }
    myfile << ",";
  }
	myfile<<std::endl;
	myfile.close();
}

void useful_macs(){
  int mac_count=0;
  if(densityA==1 && densityB==1){
    mac_count = (m*p)+(p*n);
  }

  for (i=0;i<m;i++){
    for (j=0;j<p;j++){
      A2[i*m+j] = A[(j*m)+i];
    }
  }
  // cout <<"\n Top left corner of matrix A2: \n";
  // for (i=0; i<min(m,6); i++) {
  //     for (j=0; j<min(p,6); j++) {
  //         cout <<" "<< A2[j+i*p] ;
  //     }
  //     cout<<"\n";
  // }


  // for (i=0; i<(m*n); i++){
  //   if(A[i]!=0){
  //     if(B[i]!=0){
  //       mac_count++;
  //     }
  //   }
  // }
  cout<<"\n No of Useful MAC operations are: "<<mac_count<<endl;

}

void deallocate(){
  cout << "\n Deallocating memory" << endl;
  delete[] A;
  delete[] B;
  delete[] C;
  cout<<"\n C++ file Executed. \n"<<endl;
}

void main() {
    //cout <<"\n This example computes real matrix C=alpha*A*B+beta*C using "<<endl<<" Intel(R) MKL function dgemm, where A, B, and  C are matrices and "<<endl<<" alpha and beta are double precision scalars\n"<<endl;
    read_input();
    srand(time(NULL));
    double total_Time=0.0;
    init();
    auto start = std::chrono::system_clock::now();
    matmul();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> itime = (end - start);
    total_Time += itime.count();
    std::cout << " Time for Multiplication: " << total_Time << endl;
    //print_output();
    write_output();
    write_output_for_wrapper();
    useful_macs();
    deallocate();
}
