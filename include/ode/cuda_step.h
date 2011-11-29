#ifndef _CUDA_ODE_STEP_H_
#define _CUDA_ODE_STEP_H_

#include <ode/common.h>

#ifdef __cplusplus
extern "C" {
#endif

ODE_API void cuda_dInternalStepIsland_x1 (dxWorld *world, dxBody * const *body, 
int nb, dxJoint * const *_joint, int nj, dReal stepsize);

ODE_API void dInternalStepIsland_x2 (dxWorld *world, dxBody * const *body, 
int nb, dxJoint * const *_joint, int nj, dReal stepsize)

#ifdef __cplusplus
}
#endif
#endif
