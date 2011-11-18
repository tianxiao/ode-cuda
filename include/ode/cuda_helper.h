#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_

#include <ode/common.h>

#ifdef __cplusplus
extern "C" {
#endif

ODE_API void cuda_testMemcpy();

ODE_API dReal *cuda_copyToDevice(dReal *a, int n);
ODE_API dReal *cuda_copyFromDevice(dReal *dev_a, dReal *a, int n);
ODE_API void cuda_freeFromDevice(dReal *dev_a);
ODE_API dReal *cuda_makeOnDevice(int n);

#ifdef __cplusplus
}
#endif
#endif

