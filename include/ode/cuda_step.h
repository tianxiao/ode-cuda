#ifndef _CUDA_ODE_STEP_H_
#define _CUDA_ODE_STEP_H_

#include <ode/common.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*dstepper_fn_t) (dxWorld *world, dxBody * const *body, int nb, dxJoint * const *_joint, int nj, dReal stepsize);

	ODE_API void cuda_dxProcessIslands(dxWorld *world, dxBody *cuda_body, dReal stepsize, dstepper_fn_t stepper);
ODE_API void cuda_dInternalStepIsland_x1 (dxWorld *world, dxBody *cuda_body, int nb, dxJoint * *_joint, int nj, dReal stepsize);

#ifdef __cplusplus
}
#endif
#endif
