#include <stdio.h>
#include <ode/cuda_demo_helper.h>

ODE_API void printMatrix(char *name, dReal *a, int h, int w)
{
	printf("%s:\n", name);
	for (int row=0; row<h; row++) {
		for (int col=0; col<w; col++)
			printf("%d, ", (int) a[row*w+col]);
		printf("\n");
	}
	printf("\n");
}

ODE_API void makeIdMatrix(dReal *a, int s, int n)
{
	for (int row=0; row<s; row++) {
		for (int col=0; col<s; col++) {
			if (row==col) { a[row*s+col] = n; }
			else { a[row*s+col] = 0; }
		}
	}
}

