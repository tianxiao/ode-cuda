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

ODE_API dxBody *cuda_copyBodiesToDevice(dxBody *cuda_body, dxBody **body, int NUM);
ODE_API dxBody *cuda_copyBodiesToDevice2(dxBody *cuda_body, dxWorld *world, int NUM);
ODE_API dxBody **cuda_copyBodiesFromDevice(dxBody **body, dxBody *cuda_body, int NUM, dxBody *b_buff);
ODE_API dxBody **cuda_copyBodiesFromDevice2(dxWorld *world, dxBody *cuda_body, int NUM, dxBody *b_buff);
ODE_API dxBody *cuda_initBodiesOnDevice(int NUM);
ODE_API void cuda_free(dxBody *ptr);

#ifdef __cplusplus
}
#endif
#endif

