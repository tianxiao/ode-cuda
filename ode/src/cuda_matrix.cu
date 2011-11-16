#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <ode/common.h>
#include <ode/cuda_matrix.h>
#include "util.h"
#include "config.h"

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err));
		exit(EXIT_FAILURE);
	}                         
}

void cuda_testMemcpy()
{
	float *a_h, *b_h;	// pointers to host memory
	float *a_d, *b_d;	// pointers to device memory
	int N = 14;
	int i;
	// allocate arrays on host
	a_h = (float*) malloc(sizeof(float)*N);
	b_h = (float*) malloc(sizeof(float)*N);
	// allocate arrays on device
	cudaMalloc((void**) &a_d, sizeof(float)*N);
	cudaMalloc((void**) &b_d, sizeof(float)*N);
	// initialize host data
	for (i=0; i<N; i++) {
		a_h[i] = 10.f+i;
		b_h[i] = 0.f;
	}
	// send data from host to device: a_h to a_d
	cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
	// copy data within device: a_d to b_d
	cudaMemcpy(b_d, a_d, sizeof(float)*N, cudaMemcpyDeviceToDevice);
	// retrieve data from device: b_d to b_h
	cudaMemcpy(b_h, b_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	// check result
	for (i=0; i<N; i++)
		assert(a_h[i] == b_h[i]);
	// cleanup
	free(a_h); free(b_h);
	cudaFree(a_d); cudaFree(b_d);
}

__global__ void setzero(dReal *a, int n)
{
	int tid = blockIdx.x;
	if(tid < n)
		a[tid] = 0;
}

void cuda_dSetZero(dReal *a, int n)
{
	dReal *dev_a; 

	// allocate memory on GPU
	cudaMalloc((void**) &dev_a, n*sizeof(dReal));
	checkCUDAError("malloc");

	//copy array from CPU to GPU (not necessary)
	cudaMemcpy(dev_a, a, n*sizeof(dReal), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy");

	//fill array with 0 on the gpu
	setzero<<<n,1>>>(dev_a, n);

	//copy array of 0's 'a' from GPU to CPU
	cudaMemcpy(a, dev_a, n*sizeof(dReal), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy");

	cudaFree(dev_a);		
}

