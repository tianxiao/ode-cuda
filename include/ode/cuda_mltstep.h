#ifndef _CUDA_MLTSTEP_H_
#define _CUDA_MLTSTEP_H_

#include <ode/common.h>


#ifdef __cplusplus
extern "C" {
#endif

ODE_API void mltstep_dInternalStepIsland_x1 (dxWorld *world, dxBody * const *body, int nb, dxJoint * const *_joint, int nj, dReal stepsize);

#ifdef __cplusplus
}
#endif

#endif
