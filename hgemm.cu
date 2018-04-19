#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.h"

using namespace std;

int main(int argc, char ** argv){

  int max_m_k_n = 2028;
  int repeats = 10;
  int verbose = 1;

  cout << "\nrunning cublasHgemm test\n" << endl;
  
  if(verbose) 
    cout << "running with" 
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
  
  float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  
  CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
  CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
  CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);
    
	__half *d_A, *d_B, *d_C;
  checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(__half)));
  checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(__half)));
  checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(__half)));
  
  for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
    d_A[i] = approx_float_to_half(h_A[i]);
	  d_B[i] = approx_float_to_half(h_B[i]);
	  d_C[i] = approx_float_to_half(h_C[i]);
  }
  
  int lda, ldb, ldc, m, n, k;
  const __half alf = approx_float_to_half(1.0);
  const __half bet = approx_float_to_half(0.0);
  const __half *alpha = &alf;
  const __half *beta = &bet;

  cout << "begin testing: \n";
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int i = 0; i < 1; i++){
    double sum = 0.0;

    for(int rep = 0; rep < repeats; rep++){
      cudaEventRecord(start, 0);
    	  m=256;
        n=256;
        k=256;
    	  lda = m;
    	  ldb = k;
    	  ldc = m;

      	stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc); 

      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      if(stat != CUBLAS_STATUS_SUCCESS){
      	cerr << "cublasSgemmBatched failed" << endl;
      	exit(1);
      }

      assert(!cudaGetLastError());
      
      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      elapsed /= 1000.0f;
      sum += elapsed;
    }

    cout << "float16; " << " average: " << sum/repeats << " s "<< endl;
  }

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
      
  return 0;
}
