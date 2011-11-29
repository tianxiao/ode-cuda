// cuda_step.cu

#include "objects.h"
#include "joints/joint.h"
#include <ode/odeconfig.h>
#include "config.h"
#include <ode/odemath.h>
#include <ode/rotation.h>
#include <ode/timer.h>
#include <ode/error.h>
#include <ode/matrix.h>
#include "lcp.h"
#include "util.h"

#include <ode/cuda_step.h>
#include <cuda.h>
#include <ode/cuda_helper.h>
#include <ode/cuda_matrix.h>

ODE_API void cuda_dxProcessIslands (dxWorld *world, dReal stepsize, dstepper_fn_t cuda_stepper)
{
	int cuda_bodies_count = 0;
	dxBody *cuda_bodies;
	cudaMalloc((void**) &cuda_bodies, sizeof(dxBody)*world->nb);
	dxBody *bb;
	for (bb=world->firstbody;bb;bb=(dxBody*)bb->next)
		cudaMemcpy(cuda_bodies+sizeof(dxBody)*cuda_bodies_count++, bb, sizeof(dxBody), cudaMemcpyHostToDevice);
    cuda_stepper (world,cuda_bodies,cuda_bodies_count,NULL,NULL,stepsize);
}
