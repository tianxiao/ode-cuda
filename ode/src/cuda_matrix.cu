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

template <int BLOCK_SIZE> __global__ void MatMulKernel0(cuda_Matrix C, cuda_Matrix A, cuda_Matrix B)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	cuda_Matrix C_sub = GetSubMatrix<BLOCK_SIZE>(C, blockRow, blockCol);

	dReal C_val = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < ((C.width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
		cuda_Matrix A_sub = GetSubMatrix<BLOCK_SIZE>(A, blockRow, m);
		cuda_Matrix B_sub = GetSubMatrix<BLOCK_SIZE>(B, m, blockCol);
		__shared__ dReal As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ dReal Bs[BLOCK_SIZE][BLOCK_SIZE];
		if (BLOCK_SIZE * blockRow + row < A.height && BLOCK_SIZE * m + col < A.width) {
			As[row][col] = GetElement(A_sub, row, col);
			cuPrintf("A row: %d\n", row);
			cuPrintf("A col: %d\n", col);
		} else {
			As[row][col] = 0;
			cuPrintf("\t\t\tA row: %d\n", row);
			cuPrintf("\t\t\tA col: %d\n", col);
		}
		__syncthreads();
		if (BLOCK_SIZE * m + row < B.height && BLOCK_SIZE * blockCol + col < B.width) {
			Bs[row][col] = GetElement(B_sub, row, col);
			cuPrintf("B row: %d\n", row);
			cuPrintf("B col: %d\n", col);
		} else {
			Bs[row][col] = 0;
			cuPrintf("\t\t\tB row: %d\n", row);
			cuPrintf("\t\t\tB col: %d\n", col);
		}
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			C_val += As[row][e] * Bs[e][col];
			__syncthreads();
		}
		__syncthreads();
	}
	if (BLOCK_SIZE * blockRow + row < C.height && BLOCK_SIZE * blockCol + col < C.width) {
		SetElement(C_sub, row, col, C_val);
	}
}

template <int BLOCK_SIZE> __global__ void MatMulKernel1(cuda_Matrix C, cuda_Matrix A, cuda_Matrix B)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	cuda_Matrix C_sub = GetSubMatrix<BLOCK_SIZE>(C, blockRow, blockCol);

	dReal C_val = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < ((C.width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
		cuda_Matrix A_sub = GetSubMatrix<BLOCK_SIZE>(A, blockRow, m);
		cuda_Matrix B_sub = GetSubMatrix<BLOCK_SIZE>(B, m, blockCol);
		__shared__ dReal As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ dReal Bs[BLOCK_SIZE][BLOCK_SIZE];
		if (BLOCK_SIZE * blockRow + row < A.height && BLOCK_SIZE * m + col < A.width) {
			As[row][col] = GetElement(A_sub, row, col);
			cuPrintf("A row: %d\n", row);
			cuPrintf("A col: %d\n", col);
		} else {
			As[row][col] = 0;
			cuPrintf("\t\t\tA row: %d\n", row);
			cuPrintf("\t\t\tA col: %d\n", col);
		}
		__syncthreads();
		if (BLOCK_SIZE * m + row < B.height && BLOCK_SIZE * blockCol + col < B.width) {
			Bs[row][col] = GetElement(B_sub, row, col);
			cuPrintf("B row: %d\n", row);
			cuPrintf("B col: %d\n", col);
		} else {
			Bs[row][col] = 0;
			cuPrintf("\t\t\tB row: %d\n", row);
			cuPrintf("\t\t\tB col: %d\n", col);
		}
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			C_val += As[row][e] * Bs[e][col];
			__syncthreads();
		}
		__syncthreads();
	}
	if (BLOCK_SIZE * blockRow + row < C.height && BLOCK_SIZE * blockCol + col < C.width) {
		SetElement(C_sub, row, col, C_val);
	}
}

template <int BLOCK_SIZE> __global__ void MatMulKernel2(cuda_Matrix C, cuda_Matrix A, cuda_Matrix B)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	cuda_Matrix C_sub = GetSubMatrix<BLOCK_SIZE>(C, blockRow, blockCol);

	dReal C_val = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < ((C.width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
		cuda_Matrix A_sub = GetSubMatrix<BLOCK_SIZE>(A, blockRow, m);
		cuda_Matrix B_sub = GetSubMatrix<BLOCK_SIZE>(B, m, blockCol);
		__shared__ dReal As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ dReal Bs[BLOCK_SIZE][BLOCK_SIZE];
		if (BLOCK_SIZE * blockRow + row < A.height && BLOCK_SIZE * m + col < A.width) {
			As[row][col] = GetElement(A_sub, row, col);
			cuPrintf("A row: %d\n", row);
			cuPrintf("A col: %d\n", col);
		} else {
			As[row][col] = 0;
			cuPrintf("\t\t\tA row: %d\n", row);
			cuPrintf("\t\t\tA col: %d\n", col);
		}
		__syncthreads();
		if (BLOCK_SIZE * m + row < B.height && BLOCK_SIZE * blockCol + col < B.width) {
			Bs[row][col] = GetElement(B_sub, row, col);
			cuPrintf("B row: %d\n", row);
			cuPrintf("B col: %d\n", col);
		} else {
			Bs[row][col] = 0;
			cuPrintf("\t\t\tB row: %d\n", row);
			cuPrintf("\t\t\tB col: %d\n", col);
		}
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			C_val += As[row][e] * Bs[e][col];
			__syncthreads();
		}
		__syncthreads();
	}
	if (BLOCK_SIZE * blockRow + row < C.height && BLOCK_SIZE * blockCol + col < C.width) {
		SetElement(C_sub, row, col, C_val);
	}
}

ODE_API void cuda_dMultiply0(dReal *dev_A, dReal *dev_B, dReal *dev_C, int p, int q, int r)
{
	const int block_size = 8;
	printf("cuda_dMultiply0\n");

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

	printf("Launching MatMulKernel0\n");
	printf("\tB.width: %d\n\tA.height: %d\n", B.width, A.height);
	printf("\tBlock size: %d\n", block_size);

	dim3 dimBlock(block_size, block_size);
	printf("\tdimBlock.x: %d\n\tdimBlock.y: %d\n", dimBlock.x, dimBlock.y);

	dim3 dimGrid((B.width + (block_size - 1)) / dimBlock.x, (A.height + (block_size - 1)) / dimBlock.y);
	//dim3 dimGrid((B.width + dimBlock.x) / dimBlock.x, (A.height + dimBlock.y) / dimBlock.y);
	printf("\tGrid.x: %d\n\tGrid.y: %d\n", dimGrid.x, dimGrid.y);

	//cudaPrintfInit();

	MatMulKernel0<block_size><<<dimGrid, dimBlock>>>(C, A, B);

	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();
}

ODE_API void cuda_dMultiply1(dReal *dev_C, dReal *dev_A, dReal *dev_B, int p, int q, int r)
{
	const int block_size = 8;
	printf("cuda_dMultiply1\n");

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

	printf("Launching MatMulKernel1\n");
	printf("\tB.width: %d\n\tA.height: %d\n", B.width, A.height);
	printf("\tBlock size: %d\n", block_size);

	dim3 dimBlock(block_size, block_size);
	printf("\tdimBlock.x: %d\n\tdimBlock.y: %d\n", dimBlock.x, dimBlock.y);
	dim3 dimGrid((B.width + (block_size - 1)) / dimBlock.x, (A.height + (block_size - 1)) / dimBlock.y);
	printf("\tGrid.x: %d\n\tGrid.y: %d\n", dimGrid.x, dimGrid.y);

	MatMulKernel1<block_size><<<dimGrid, dimBlock>>>(C, A, B);
}

ODE_API void cuda_dMultiply2(dReal *dev_A, dReal *dev_B, dReal *dev_C, int p, int q, int r)
{
	const int block_size = 8;
	printf("cuda_dMultiply2\n");

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

	printf("Launching MatMulKernel2\n");
	printf("\tB.width: %d\n\tA.height: %d\n", B.width, A.height);
	printf("\tBlock size: %d\n", block_size);

	dim3 dimBlock(block_size, block_size);
	printf("\tdimBlock.x: %d\n\tdimBlock.y: %d\n", dimBlock.x, dimBlock.y);
	dim3 dimGrid((B.width + (block_size - 1)) / dimBlock.x, (A.height + (block_size - 1)) / dimBlock.y);
	printf("\tGrid.x: %d\n\tGrid.y: %d\n", dimGrid.x, dimGrid.y);

	MatMulKernel2<block_size><<<dimGrid, dimBlock>>>(C, A, B);
}

