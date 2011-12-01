#include <ode/ode.h>

void fat_matrix(int dim)
{
	dReal *A = (dReal *) malloc(sizeof(dReal)*dim*dim);
	dReal *B = (dReal *) malloc(sizeof(dReal)*dim*dim);
	dReal *C = (dReal *) malloc(sizeof(dReal)*dim*dim);
	dSetValue(A, 1, dim * dim);
	dSetValue(B, 2, dim * dim);
	dMultiply0(C, A, B, dim, dim, dim);
	dMultiply1(C, A, B, dim, dim, dim);
	dMultiply2(C, A, B, dim, dim, dim);
	free(A);
	free(B);
	free(C);
}

void cuda_fat_matrix(int dim)
{
		//dReal *host_A = (dReal *) malloc(sizeof(dReal)*dim*dim);
		//dReal *host_B = (dReal *) malloc(sizeof(dReal)*dim*dim);
		//dReal *host_C = (dReal *) malloc(sizeof(dReal)*dim*dim);
		//dSetValue(host_A, 1, dim * dim);
		//dSetValue(host_B, 2, dim * dim);
		//dSetZero(host_C, dim * dim);
	dReal *A = cuda_makeOnDevice(dim*dim);
	dReal *B = cuda_makeOnDevice(dim*dim);
	dReal *C = cuda_makeOnDevice(dim*dim);
		//dReal *A = cuda_copyToDevice(host_A, dim * dim);
		//dReal *B = cuda_copyToDevice(host_B, dim * dim);
		//dReal *C = cuda_copyToDevice(host_C, dim * dim);
	cuda_dSetValue(A, 1, dim * dim);
	cuda_dSetValue(B, 2, dim * dim);
	cuda_dMultiply0(C, A, B, dim, dim, dim);
	cuda_dMultiply1(C, A, B, dim, dim, dim);
	cuda_dMultiply2(C, A, B, dim, dim, dim);
		//cuda_copyFromDevice(A, host_A, dim * dim);
		//cuda_copyFromDevice(B, host_B, dim * dim);
		//cuda_copyFromDevice(C, host_C, dim * dim);
		//free(host_A);
		//free(host_B);
		//free(host_C);
	cuda_freeFromDevice(A);
	cuda_freeFromDevice(B);
	cuda_freeFromDevice(C);
}

int main(int argc, char *argv[])
{
	int dim;
	if (argc < 3 || (dim = atoi(argv[2])) <= 0) {
		fprintf(stderr, "Usage: %s {c|o} DIM\n", argv[0]);
		exit(1);
	}
	if (argv[1][0] == 'c') {
		cuda_fat_matrix(dim);
	} else {
		fat_matrix(dim);
	}
	return 0;
}

