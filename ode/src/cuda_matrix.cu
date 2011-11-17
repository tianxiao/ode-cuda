#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <ode/common.h>
#include <ode/cuda_matrix.h>
#include "util.h"
#include "config.h"

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((dReal*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((dReal*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

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

//gets the row column element of A^T
__device__ dReal GetTransposeElement(cuda_Matrix A, int row, int col)
{
	return A.elements[col * A.stride + row];
}

__device__ void SetElement(cuda_Matrix A, int row, int col, dReal val)
{
	A.elements[row * A.stride + col] = val;
}

template <int BLOCK_SIZE> __device__ cuda_Matrix GetSubMatrix(cuda_Matrix A, int row, int col)
{
	cuda_Matrix A_sub;
	A_sub.width = BLOCK_SIZE;
	A_sub.height = BLOCK_SIZE;
	A_sub.stride = A.stride;
	A_sub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return A_sub;
}

template <int BLOCK_SIZE> __global__ void MatMulKernel(cuda_Matrix A, cuda_Matrix B, cuda_Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	cuda_Matrix C_sub = GetSubMatrix<BLOCK_SIZE>(C, blockRow, blockCol);
	
	dReal C_val = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		cuda_Matrix A_sub = GetSubMatrix<BLOCK_SIZE>(A, blockRow, m);
		cuda_Matrix B_sub = GetSubMatrix<BLOCK_SIZE>(B, m, blockRow);
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

template <int BLOCK_SIZE> __global__ void
matrixMul( dReal* C, dReal* A, dReal* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    dReal Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ dReal As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ dReal Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void cuda_dMultiply0(dReal *dev_A, dReal *dev_B, dReal *dev_C, int p, int q, int r)
{
	const int block_size = 4;

	cuda_Matrix A;
	A.width = r;
	A.height = p;
	A.stride = r;
	A.elements = dev_A;

	cuda_Matrix B;
	B.width = r;
	B.height = p;
	B.stride = r;
	B.elements = dev_B;

	cuda_Matrix C;
	C.width = r;
	C.height = q;
	C.stride = r;
	C.elements = dev_C;

	dim3 dimBlock(block_size, block_size);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<2><<<dimGrid, dimBlock>>>(B, C, A);
	/*
	unsigned int uiWC, uiHC;
	uiWC = 2 * block_size;
	uiHC = 4 * block_size;
	dim3 threads(block_size, block_size);
	dim3 grid(uiWC / threads.x, uiHC / threads.y);
	if (block_size == 4) {
		matrixMul<4><<< grid, threads >>>(dev_B, dev_C, dev_A, r, r);
	} else {
		matrixMul<4><<< grid, threads >>>(dev_B, dev_C, dev_A, r, r);
	}
	*/
}

void cuda_dMultiply1(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}

void cuda_dMultiply2(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}
