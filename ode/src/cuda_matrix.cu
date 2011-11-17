#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <ode/common.h>
#include <ode/cuda_matrix.h>
#include "util.h"
#include "config.h"

#define BLOCK_SIZE 4

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
	cuda_checkError("memcpy h to d");
	return dev_a;
}

dReal *cuda_copyFromDevice(dReal *dev_a, dReal *a, int n)
{
	cudaMemcpy(a, dev_a, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cuda_checkError("memcpy d to h");
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
	cuda_checkError("dSetZero2; memcpy h to d");

	//fill array with 0 on the gpu
	setzero<<<n,1>>>(dev_a, n);

	//copy array of 0's 'a' from GPU to CPU
	cudaMemcpy(a, dev_a, n*sizeof(dReal), cudaMemcpyDeviceToHost);
	cuda_checkError("dSetZero2; memcpy d to h");

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

typedef struct {
	int width;
	int height;
	int stride;
	dReal *elements;
} cuda_Matrix;

__device__ dReal GetElement(cuda_Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

__device__ void SetElement(cuda_Matrix A, int row, int col, dReal val)
{
	A.elements[row * A.stride + col] = val;
}

__device__ cuda_Matrix GetSubMatrix(cuda_Matrix A, int row, int col)
{
	cuda_Matrix A_sub;
	A_sub.width = BLOCK_SIZE;
	A_sub.height = BLOCK_SIZE;
	A_sub.stride = A.stride;
	A_sub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return A_sub;
}

__global__ void MatMulKernel(cuda_Matrix A, cuda_Matrix B, cuda_Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	cuda_Matrix C_sub = GetSubMatrix(C, blockRow, blockCol);
	
	dReal C_val = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		cuda_Matrix A_sub = GetSubMatrix(A, blockRow, m);
		cuda_Matrix B_sub = GetSubMatrix(B, m, blockRow);
		__shared__ dReal As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ dReal Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[row][col] = GetElement(A_sub, row, col);
		Bs[row][col] = GetElement(B_sub, row, col);
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			C_val += As[row][e] * Bs[e][col];
		}
		__syncthreads();
	}
	SetElement(C_sub, row, col, C_val);
}

void cuda_dMultiply0(dReal *dev_A, dReal *dev_B, dReal *dev_C, int p, int q, int r)
{
	cuda_Matrix A;
	A.width = r;
	A.height = p;
	A.stride = r;
	A.elements = dev_A;

	cuda_Matrix B;
	B.width = r;
	B.height = p;
	B.stride = q;
	B.elements = dev_B;

	cuda_Matrix C;
	C.width = r;
	C.height = q;
	C.stride = r;
	C.elements = dev_C;

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(B, C, A);
}

void cuda_dMultiply1(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}

void cuda_dMultiply2(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}
