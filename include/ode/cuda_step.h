#ifndef _CUDA_ODE_STEP_H_
#define _CUDA_ODE_STEP_H_

#include <ode/common.h>

#ifdef __cplusplus
extern "C" {
#endif

ODE_API void cuda_step (dxWorld *world, dxBody * const *body, int nb, dReal stepsize);

#ifdef __cplusplus
}
#endif
#endif
