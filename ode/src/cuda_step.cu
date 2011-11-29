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



#define ALLOCA(t,v,s)                           \
  Auto<t> v( dALLOCA16(s) );


//****************************************************************************
// the slow, but sure way
// note that this does not do any joint feedback!

// given lists of bodies and joints that form an island, perform a first
// order timestep.
//
// `body' is the body array, `nb' is the size of the array.
// `_joint' is the body array, `nj' is the size of the array.

__global__ void cuda_step(dxWorld *world, dxBody * const *body, int nb, dReal stepsize)
{
  int i,j,k;
  int n6 = 6*nb;

  // for all bodies, compute the inertia tensor and its inverse in the global
  // frame, and compute the rotational force and add it to the torque
  // accumulator.
  // @@@ check computation of rotational force.
  ALLOCA(dReal,I,3*nb*4*sizeof(dReal));
  ALLOCA(dReal,invI,3*nb*4*sizeof(dReal));

  //dSetZero (I,3*nb*4);
  //dSetZero (invI,3*nb*4);
  for (i=0; i<nb; i++) {
    dReal tmp[12];
    // compute inertia tensor in global frame
    dMULTIPLY2_333 (tmp,body[i]->mass.I,body[i]->posr.R);
    dMULTIPLY0_333 (I+i*12,body[i]->posr.R,tmp);
    // compute inverse inertia tensor in global frame
    dMULTIPLY2_333 (tmp,body[i]->invI,body[i]->posr.R);
    dMULTIPLY0_333 (invI+i*12,body[i]->posr.R,tmp);
    // compute rotational force
    dMULTIPLY0_331 (tmp,I+i*12,body[i]->avel);
    dCROSS (body[i]->tacc,-=,body[i]->avel,tmp);
  }

  // add the gravity force to all bodies
  for (i=0; i<nb; i++) {
    if ((body[i]->flags & dxBodyNoGravity)==0) {
      body[i]->facc[0] += body[i]->mass.mass * world->gravity[0];
      body[i]->facc[1] += body[i]->mass.mass * world->gravity[1];
      body[i]->facc[2] += body[i]->mass.mass * world->gravity[2];
    }
  }
  // create (6*nb,6*nb) inverse mass matrix `invM', and fill it with mass
  // parameters

  int nskip = dPAD (n6);
  ALLOCA(dReal, invM, n6*nskip*sizeof(dReal));
  
  dSetZero (invM,n6*nskip);
  for (i=0; i<nb; i++) {
    dReal *MM = invM+(i*6)*nskip+(i*6);
    MM[0] = body[i]->invMass;
    MM[nskip+1] = body[i]->invMass;
    MM[2*nskip+2] = body[i]->invMass;
    MM += 3*nskip+3;
    for (j=0; j<3; j++) for (k=0; k<3; k++) {
      MM[j*nskip+k] = invI[i*12+j*4+k];
    }
  }

  // assemble some body vectors: fe = external forces, v = velocities
  ALLOCA(dReal,fe,n6*sizeof(dReal));
  ALLOCA(dReal,v,n6*sizeof(dReal));

  //dSetZero (fe,n6);
  //dSetZero (v,n6);
  for (i=0; i<nb; i++) {
    for (j=0; j<3; j++) fe[i*6+j] = body[i]->facc[j];
    for (j=0; j<3; j++) fe[i*6+3+j] = body[i]->tacc[j];
    for (j=0; j<3; j++) v[i*6+j] = body[i]->lvel[j];
    for (j=0; j<3; j++) v[i*6+3+j] = body[i]->avel[j];
  }

  // this will be set to the velocity update
  ALLOCA(dReal,vnew,n6*sizeof(dReal));
  dSetZero (vnew,n6);

  // no constraints
  dMultiply0 (vnew,invM,fe,n6,n6,1);
  for (i=0; i<n6; i++) vnew[i] = v[i] + stepsize*vnew[i];

  // apply the velocity update to the bodies
  for (i=0; i<nb; i++) {
    for (j=0; j<3; j++) body[i]->lvel[j] = vnew[i*6+j];
    for (j=0; j<3; j++) body[i]->avel[j] = vnew[i*6+3+j];
  }

  // update the position and orientation from the new linear/angular velocity
  // (over the given timestep)
  for (i=0; i<nb; i++) dxStepBody (body[i],stepsize);

  // zero all force accumulators
  for (i=0; i<nb; i++) {
    body[i]->facc[0] = 0;
    body[i]->facc[1] = 0;
    body[i]->facc[2] = 0;
    body[i]->facc[3] = 0;
    body[i]->tacc[0] = 0;
    body[i]->tacc[1] = 0;
    body[i]->tacc[2] = 0;
    body[i]->tacc[3] = 0;
  }
}
