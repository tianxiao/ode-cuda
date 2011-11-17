#ifndef _CUDA_ODE_MATRIX_H_
#define _CUDA_ODE_MATRIX_H_

#include <ode/common.h>

#ifdef __cplusplus
extern "C" {
#endif

ODE_API void cuda_testMemcpy();
ODE_API void cuda_dSetZero(dReal *dev_a, int n);
ODE_API void cuda_dSetValue(dReal *dev_a, int n, dReal value);
ODE_API void cuda_dMultiply0(dReal *dev_A, dReal *dev_B, dReal *dev_c, int p, int q, int r);
ODE_API void cuda_dMultiply1(dReal *dev_A, dReal *dev_B, dReal *dev_c, int p, int q, int r);
ODE_API void cuda_dMultiply2(dReal *dev_A, dReal *dev_B, dReal *dev_c, int p, int q, int r);

ODE_API dReal *cuda_copyToDevice(dReal *a, int n);
ODE_API dReal *cuda_copyFromDevice(dReal *dev_a, dReal *a, int n);
ODE_API void cuda_freeFromDevice(dReal *dev_a);

#ifdef __cplusplus
}
#endif
#endif
