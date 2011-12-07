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
//		printf("\tbody[%d]->posr.pos[0] = %f\n", i, body[i]->posr.pos[0]);
	}
	cuda_checkError("memcpy bodies h to d");
	return cuda_body;
}

ODE_API dxBody *cuda_copyBodiesToDevice2(dxBody *cuda_body, dxWorld *world, int NUM)
{
	dxBody *b;
	int i=0;
	for (b=world->firstbody;b;b=(dxBody*)b->next) {
		cudaMemcpy(cuda_body+(++i), b, sizeof(dxBody), cudaMemcpyHostToDevice);
//		printf("\t%d b.posr.pos[0] = %f\n", i, b->posr.pos[0]);
//		printf("%f\n", b->posr.pos[0]);
	}
	cuda_checkError("memcpy bodies h to d 2");
	return cuda_body;
}

ODE_API dxBody **cuda_copyBodiesFromDevice(dxWorld *world, dxBody *cuda_body, int NUM, dxBody *b_buff)
{
//	printf("Copy Bodies From Device");
	int i=0;
	cudaMemcpy(b_buff, cuda_body, sizeof(dxBody)*NUM, cudaMemcpyDeviceToHost);
	cuda_checkError("memcpy bodies from device d to h");
	printf("GOt HERE\n");
/*	for (i=0;i<NUM;i++) {
		//dxBody *b = (dxBody *) malloc(sizeof(dxBody));
		//cudaMemcpy(b, cuda_body+i, sizeof(dxBody), cudaMemcpyDeviceToHost);
		//body[i] = b;
		cudaMemcpy(b[i], cuda_body, sizeof(dxBody), cudaMemcpyDeviceToHost);
	}*/
	dxBody *b;
	int x;
	for (b=world->firstbody;b;b=(dxBody*)b->next) {
		b->flags = b_buff[i].flags;
		b->geom = b_buff[i].geom;
		b->mass = b_buff[i].mass;
		for (x=0;x<3*3;x++) {
			b->invI[x] = b_buff[i].invI[x];
		}
		b->invMass = b_buff[i].invMass;
		b->posr = b_buff[i].posr;
		b->q[0] = b_buff[i].q[0];
		b->q[1] = b_buff[i].q[1];
		b->q[2] = b_buff[i].q[2];
		b->q[3] = b_buff[i].q[3];
		b->lvel[0] = b_buff[i].lvel[0];
		b->lvel[1] = b_buff[i].lvel[1];
		b->lvel[2] = b_buff[i].lvel[2];
		b->avel[0] = b_buff[i].avel[0];
		b->avel[1] = b_buff[i].avel[1];
		b->avel[2] = b_buff[i].avel[2];
		b->facc[0] = b_buff[i].facc[0];
		b->facc[1] = b_buff[i].facc[1];
		b->facc[2] = b_buff[i].facc[2];
		b->tacc[0] = b_buff[i].tacc[0];
		b->tacc[1] = b_buff[i].tacc[1];
		b->tacc[2] = b_buff[i].tacc[2];
		b->finite_rot_axis[0] = b_buff[i].finite_rot_axis[0];
		b->finite_rot_axis[1] = b_buff[i].finite_rot_axis[1];
		b->finite_rot_axis[2] = b_buff[i].finite_rot_axis[2];
		b->adis = b_buff[i].adis;
		b->adis_timeleft = b_buff[i].adis_timeleft;
		b->adis_stepsleft = b_buff[i].adis_stepsleft;
		b->average_counter = b_buff[i].average_counter;
		b->average_ready = b_buff[i].average_ready;
		b->dampingp = b_buff[i].dampingp;
		b->max_angular_speed = b_buff[i].max_angular_speed;
		i++;
	}
	return NULL;
}

ODE_API dxBody *cuda_initBodiesOnDevice(int NUM)
{
//	printf("Init %i Bodies\n", sizeof(dxBody));
	dxBody *cuda_body;
	cudaMalloc((void**) &cuda_body, sizeof(dxBody)*NUM);
	cuda_checkError("malloc");
	return cuda_body;
}

ODE_API void cuda_free(dxBody *ptr)
{
	cudaFree(ptr);
}

