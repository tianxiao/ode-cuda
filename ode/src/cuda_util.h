#ifndef _CUDA_ODE_UTIL_H_
#define _CUDA_ODE_UTIL_H_

#include "objects.h"

#ifdef __cplusplus
extern "C" {
#endif

ODE_API void cuda_dInternalHandleAutoDisabling (dxWorld *world, dReal stepsize);
ODE_API void cuda_dxStepBody (dxBody *b, dReal h);

/*typedef void (*dstepper_fn_t) (dxWorld *world, dxBody * const *body, int nb,
        dxJoint * const *_joint, int nj, dReal stepsize);*/

ODE_API void cuda_dxProcessIslands (dxWorld *world, dReal stepsize, dstepper_fn_t stepper);

#ifdef __cplusplus
}
#endif

#endif
