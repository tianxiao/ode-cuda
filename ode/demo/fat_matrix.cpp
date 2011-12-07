#include <ode/ode.h>

void fat_matrix(int p, int q, int r)
{
	dReal *A = (dReal *) malloc(sizeof(dReal)*p*q);
	dReal *B = (dReal *) malloc(sizeof(dReal)*q*r);
	dReal *C = (dReal *) malloc(sizeof(dReal)*p*r);
	dSetValue(A, 1, p*q);
	dSetValue(B, 2, q*r);
	dMultiply0(C, A, B, p, q, r);
	dMultiply1(C, A, B, p, q, r);
	dMultiply2(C, A, B, p, q, r);
	free(A);
	free(B);
	free(C);
}

void cuda_fat_matrix(int p, int q, int r)
{
		//dReal *host_A = (dReal *) malloc(sizeof(dReal)*dim*dim);
		//dReal *host_B = (dReal *) malloc(sizeof(dReal)*dim*dim);
		//dReal *host_C = (dReal *) malloc(sizeof(dReal)*dim*dim);
		//dSetValue(host_A, 1, dim * dim);
		//dSetValue(host_B, 2, dim * dim);
		//dSetZero(host_C, dim * dim);
	dReal *A = cuda_makeOnDevice(p*q);
	dReal *B = cuda_makeOnDevice(q*r);
	dReal *C = cuda_makeOnDevice(p*r);
		//dReal *A = cuda_copyToDevice(host_A, dim * dim);
		//dReal *B = cuda_copyToDevice(host_B, dim * dim);
		//dReal *C = cuda_copyToDevice(host_C, dim * dim);
	cuda_dSetValue(A, 1, p*q);
	cuda_dSetValue(B, 2, q*r);
	cuda_dMultiply0(C, A, B, p, q, r);
	cuda_dMultiply1(C, A, B, p,	q, r);
	cuda_dMultiply2(C, A, B, p, q, r);
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
	if (argc < 5 || ((p = atoi(argv[2])) <= 0) || ((q = atoi(argv[3])) <= 0) || ((r = atoi(argv[4])) <= 0)) {
		fprintf(stderr, "Usage: %s {c|o} P Q R\n", argv[0]);
		exit(1);
	}
	if (argv[1][0] == 'c') {
		cuda_fat_matrix(p,q,r);
	} else {
		fat_matrix(p,q,r);
	}
	return 0;
}

