#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <ode/common.h>
#include "objects.h"
#include <ode/cuda_helper.h>

ODE_API void cuda_checkError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err));
		exit(EXIT_FAILURE);
	}                         
}

ODE_API void cuda_testMemcpy()
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

ODE_API dReal *cuda_copyToDevice(dReal *a, int n)
{
	dReal *dev_a;
	cudaMalloc((void**) &dev_a, sizeof(dReal)*n);
	cuda_checkError("malloc");
	cudaMemcpy(dev_a, a, sizeof(dReal)*n, cudaMemcpyHostToDevice);
	cuda_checkError("memcpy h to d");
	return dev_a;
}

ODE_API dReal *cuda_copyFromDevice(dReal *dev_a, dReal *a, int n)
{
	cudaMemcpy(a, dev_a, sizeof(dReal)*n, cudaMemcpyDeviceToHost);
	cuda_checkError("memcpy d to h");
	return a;
}

ODE_API void cuda_freeFromDevice(dReal *dev_a)
{
	cudaFree(dev_a);
}

ODE_API dReal *cuda_makeOnDevice(int n)
{
	dReal *dev_a;
	cudaMalloc((void**) &dev_a, sizeof(dReal)*n);
	cuda_checkError("malloc");
	return dev_a;
}

ODE_API dxBody *cuda_copyBodiesToDevice(dxBody *cuda_body, dxBody **body, int NUM)
{
	int i;
	for (i=0;i<NUM;i++) {
		cudaMemcpy(cuda_body+i, body[i], sizeof(dxBody), cudaMemcpyHostToDevice);
		printf("%f\n", body[i]->posr.pos[0]);
	}
	cuda_checkError("memcpy h to d");
	return cuda_body;
}

ODE_API dxBody **cuda_copyBodiesFromDevice(dxBody **body, dxBody *cuda_body, int NUM)
{
	int i;
	for (i=0;i<NUM;i++) {
/*		free(body[i]);
		dxBody *b = (dxBody *) malloc(sizeof(dxBody));
		cudaMemcpy(b, cuda_body+i, sizeof(dxBody), cudaMemcpyDeviceToHost);
		body[i] = b;*/
		cudaMemcpy(body[i], cuda_body+i, sizeof(dxBody), cudaMemcpyDeviceToHost);
	}
	cuda_checkError("memcpy d to h");
	return body;
}

ODE_API dxBody *cuda_initBodiesOnDevice(int NUM)
{
	printf("%i\n", sizeof(dxBody));
	dxBody *cuda_body;
	cudaMalloc((void**) &cuda_body, sizeof(dxBody)*NUM);
	cuda_checkError("malloc");
	return cuda_body;
}

ODE_API void cuda_free(dxBody *ptr)
{
	cudaFree(ptr);
}

