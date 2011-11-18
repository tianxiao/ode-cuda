#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <ode/common.h>
#include <ode/cuda_helper.h>
#include <ode/cuda_matrix.h>
#include "cuPrintf.cu"

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

	dReal C_val = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	__shared__ dReal As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ dReal Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int m = 0; m < ((C.width + 1) / BLOCK_SIZE); ++m) {
		As[row][col] = A.elements[A.stride * BLOCK_SIZE * blockRow + BLOCK_SIZE * m + row * A.stride + col];
		Bs[row][col] = B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * blockCol + row * C.stride + col];
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			cuPrintf("e: %d", (int) C_val);
			C_val += (BLOCK_SIZE * blockRow + row < A.height && BLOCK_SIZE * blockCol + col < B.width && BLOCK_SIZE * m + e < B.height && BLOCK_SIZE * m + e < A.width) ? (As[row][e] * Bs[e][col]) : 0;
		}
	}
	__syncthreads();
	C.elements[C.stride * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol + row * C.stride + col] = C_val;
	__syncthreads();
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
	B.width = q;
	B.height = p;
	B.stride = q;
	B.elements = dev_B;

	cuda_Matrix C;
	C.width = r;
	C.height = q;
	C.stride = r;
	C.elements = dev_C;

	printf("Block size: %d\n", block_size);
	dim3 dimBlock(block_size, block_size);
	printf("dimBlock.x: %d\ndimBlock.y: %d\n", dimBlock.x, dimBlock.y);
	dim3 dimGrid((B.width + 1) / dimBlock.x + 1, (A.height + 1) / dimBlock.y);
	printf("B.width: %d\nA.height: %d\n", B.width, A.height);
	printf("Grid.x: %d\nGrid.y: %d\n", dimGrid.x, dimGrid.y);
	cudaPrintfInit();
	MatMulKernel<block_size><<<dimGrid, dimBlock>>>(B, C, A);
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	printf("\n");
}

void cuda_dMultiply1(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}

void cuda_dMultiply2(dReal *dev_A, const dReal *dev_B, const dReal *dev_c, int p, int q, int r)
{
}
