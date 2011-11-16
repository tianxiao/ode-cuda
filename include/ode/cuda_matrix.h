#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include <ode/common.h>

#ifdef __cplusplus
extern "C" {
#endif

ODE_API void cuda_dSetZero (dReal *a, int n);

#ifdef __cplusplus
}
#endif
#endif
