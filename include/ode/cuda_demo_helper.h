#ifndef _CUDA_DEMO_HELPER_H_
#define _CUDA_DEMO_HELPER_H_

#include <ode/common.h>


#ifdef __cplusplus
extern "C" {
#endif

ODE_API void printMatrix(char *name, dReal *a, int h, int w);
ODE_API void makeIdMatrix(dReal *a, int s, int n);

#ifdef __cplusplus
}
#endif

#endif
