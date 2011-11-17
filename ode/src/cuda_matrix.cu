#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <ode/common.h>
#include <ode/cuda_matrix.h>
#include "util.h"
#include "config.h"

void cuda_checkError(const char *msg)
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

dReal *cuda_copyToDevice(dReal *a, int n)
{
	dReal *dev_a;
	cudaMalloc((void**) &dev_a, sizeof(dReal)*n);
	cuda_checkError("malloc");
	cudaMemcpy(dev_a, a, sizeof(dReal)*n, cudaMemcpyHostToDevice);
	cuda_checkError("memcpy");
	return dev_a;
}

dReal *cuda_copyFromDevice(dReal *dev_a, dReal *a, int n)
{
	cudaMemcpy(a, dev_a, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cuda_checkError("memcpy");
	return a;
}

void cuda_freeFromDevice(dReal *dev_a)
{
	cudaFree(dev_a);
}

__global__ void setzero(dReal *a, int n)
{
	int tid = blockIdx.x;
	if(tid < n)
		a[tid] = 0;
}

__global__ void setvalue(dReal *a, int n, dReal value)
{
	int tid = blockIdx.x;
	if(tid < n)
		a[tid] = value;
}

void cuda_dSetZero2(dReal *a, int n)
{
	dReal *dev_a; 

	// allocate memory on GPU
	cudaMalloc((void**) &dev_a, n*sizeof(dReal));
	cuda_checkError("malloc");

	//copy array from CPU to GPU (not necessary)
	cudaMemcpy(dev_a, a, n*sizeof(dReal), cudaMemcpyHostToDevice);
	cuda_checkError("memcpy");

	//fill array with 0 on the gpu
	setzero<<<n,1>>>(dev_a, n);

	//copy array of 0's 'a' from GPU to CPU
	cudaMemcpy(a, dev_a, n*sizeof(dReal), cudaMemcpyDeviceToHost);
	cuda_checkError("memcpy");

	cudaFree(dev_a);		
}

void cuda_dSetZero(dReal *dev_a, int n)
{
	setzero<<<n,1>>>(dev_a, n);
}

void cuda_dSetValue(dReal *dev_a, int n, dReal value)
{
	setvalue<<<n,1>>>(dev_a, n, value);
}

void cuda_dMultiply0(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}

void cuda_dMultiply1(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}

void cuda_dMultiply2(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}
