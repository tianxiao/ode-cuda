/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001,2002 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of EITHER:                                  *
 *   (1) The GNU Lesser General Public License as published by the Free  *
 *       Software Foundation; either version 2.1 of the License, or (at  *
 *       your option) any later version. The text of the GNU Lesser      *
 *       General Public License is included with this library in the     *
 *       file LICENSE.TXT.                                               *
 *   (2) The BSD-style license that is included with this library in     *
 *       the file LICENSE-BSD.TXT.                                       *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

#include "objects.h"
#include "joints/joint.h"
#include <ode/odeconfig.h>
#include "config.h"
#include <ode/odemath.h>
#include <ode/rotation.h>
#include <ode/timer.h>
#include <ode/cuda_helper.h>
#include <ode/cuda_matrix.h>
#include <ode/error.h>
#include <ode/matrix.h>
#include "lcp.h"
#include "util.h"

//****************************************************************************
// misc defines

#define FAST_FACTOR
//#define TIMING

// memory allocation system
#ifdef dUSE_MALLOC_FOR_ALLOCA
unsigned int dMemoryFlag;
#define REPORT_OUT_OF_MEMORY fprintf(stderr, "Insufficient memory to complete rigid body simulation.  Results will not be accurate.\n")

#define CHECK(p)                                \
  if (!p) {                                     \
    dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;       \
    return;                                     \
  }

#define ALLOCA(t,v,s)                           \
  Auto<t> v(malloc(s));                         \
  CHECK(v)

#else // use alloca()

#define ALLOCA(t,v,s)                           \
  Auto<t> v( dALLOCA16(s) );

#endif



/* This template should work almost like std::auto_ptr
 */
template<class T>
struct Auto {
  T *p;
  Auto(void * q) :
    p(reinterpret_cast<T*>(q))
  { }

  ~Auto()
  {
#ifdef dUSE_MALLOC_FOR_ALLOCA
    free(p);
#endif
  }

  operator T*() 
  {
    return p;
  }
  T& operator[] (int i)
  {
    return p[i];
  }
private:
  // intentionally undefined, don't use this
  template<class U>
  Auto& operator=(const Auto<U>&) const;
};





//****************************************************************************
// debugging - comparison of various vectors and matrices produced by the
// slow and fast versions of the stepper.

//#define COMPARE_METHODS

#ifdef COMPARE_METHODS
#include "testing.h"
dMatrixComparison comparator;
#endif

// undef to use the fast decomposition
#define DIRECT_CHOLESKY
#undef REPORT_ERROR

//****************************************************************************
// special matrix multipliers

// this assumes the 4th and 8th rows of B and C are zero.

static void Multiply2_p8r (dReal *A, dReal *B, dReal *C,
			   int p, int r, int Askip)
{
  int i,j;
  dReal sum,*bb,*cc;
  dIASSERT (p>0 && r>0 && A && B && C);
  bb = B;
  for (i=p; i; i--) {
    cc = C;
    for (j=r; j; j--) {
      sum = bb[0]*cc[0];
      sum += bb[1]*cc[1];
		  sum += bb[2]*cc[2];
      sum += bb[4]*cc[4];
      sum += bb[5]*cc[5];
      sum += bb[6]*cc[6];
      *(A++) = sum; 
      cc += 8;
    }
    A += Askip - r;
    bb += 8;
  }
}


// this assumes the 4th and 8th rows of B and C are zero.

static void MultiplyAdd2_p8r (dReal *A, dReal *B, dReal *C,
			      int p, int r, int Askip)
{
  int i,j;
  dReal sum,*bb,*cc;
  dIASSERT (p>0 && r>0 && A && B && C);
  bb = B;
  for (i=p; i; i--) {
    cc = C;
    for (j=r; j; j--) {
      sum = bb[0]*cc[0];
      sum += bb[1]*cc[1];
      sum += bb[2]*cc[2];
      sum += bb[4]*cc[4];
      sum += bb[5]*cc[5];
      sum += bb[6]*cc[6];
      *(A++) += sum; 
      cc += 8;
    }
    A += Askip - r;
    bb += 8;
  }
}


// this assumes the 4th and 8th rows of B are zero.

static void Multiply0_p81 (dReal *A, dReal *B, dReal *C, int p)
{
  int i;
  dIASSERT (p>0 && A && B && C);
  dReal sum;
  for (i=p; i; i--) {
    sum =  B[0]*C[0];
    sum += B[1]*C[1];
    sum += B[2]*C[2];
    sum += B[4]*C[4];
    sum += B[5]*C[5];
    sum += B[6]*C[6];
    *(A++) = sum;
    B += 8;
  }
}


// this assumes the 4th and 8th rows of B are zero.

static void MultiplyAdd0_p81 (dReal *A, dReal *B, dReal *C, int p)
{
  int i;
  dIASSERT (p>0 && A && B && C);
  dReal sum;
  for (i=p; i; i--) {
    sum =  B[0]*C[0];
    sum += B[1]*C[1];
    sum += B[2]*C[2];
    sum += B[4]*C[4];
    sum += B[5]*C[5];
    sum += B[6]*C[6];
    *(A++) += sum;
    B += 8;
  }
}


// this assumes the 4th and 8th rows of B are zero.

static void MultiplyAdd1_8q1 (dReal *A, dReal *B, dReal *C, int q)
{
  int k;
  dReal sum;
  dIASSERT (q>0 && A && B && C);
  sum = 0;
  for (k=0; k<q; k++) sum += B[k*8] * C[k];
  A[0] += sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[1+k*8] * C[k];
  A[1] += sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[2+k*8] * C[k];
  A[2] += sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[4+k*8] * C[k];
  A[4] += sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[5+k*8] * C[k];
  A[5] += sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[6+k*8] * C[k];
  A[6] += sum;
}


// this assumes the 4th and 8th rows of B are zero.

static void Multiply1_8q1 (dReal *A, dReal *B, dReal *C, int q)
{
  int k;
  dReal sum;
  dIASSERT (q>0 && A && B && C);
  sum = 0;
  for (k=0; k<q; k++) sum += B[k*8] * C[k];
  A[0] = sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[1+k*8] * C[k];
  A[1] = sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[2+k*8] * C[k];
  A[2] = sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[4+k*8] * C[k];
  A[4] = sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[5+k*8] * C[k];
  A[5] = sum;
  sum = 0;
  for (k=0; k<q; k++) sum += B[6+k*8] * C[k];
  A[6] = sum;
}

//****************************************************************************
// the slow, but sure way
// note that this does not do any joint feedback!

// given lists of bodies and joints that form an island, perform a first
// order timestep.
//
// `body' is the body array, `nb' is the size of the array.
// `_joint' is the body array, `nj' is the size of the array.

void mltstep_dInternalStepIsland_x1 (dxWorld *world, dxBody * const *body, int nb,
			     dxJoint * const *_joint, int nj, dReal stepsize)
{
  int i,j,k;
  int n6 = 6*nb;
  dReal *dev_invM, *dev_J, *dev_JinvM, *dev_A;

#ifdef TIMING
  dTimerStart("preprocessing");
#endif

  // number all bodies in the body list - set their tag values
  for (i=0; i<nb; i++) body[i]->tag = i;

  // make a local copy of the joint array, because we might want to modify it.
  // (the "dxJoint *const*" declaration says we're allowed to modify the joints
  // but not the joint array, because the caller might need it unchanged).
  ALLOCA(dxJoint*,joint,nj*sizeof(dxJoint*));
  memcpy (joint,_joint,nj * sizeof(dxJoint*));

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

  // get m = total constraint dimension, nub = number of unbounded variables.
  // create constraint offset array and number-of-rows array for all joints.
  // the constraints are re-ordered as follows: the purely unbounded
  // constraints, the mixed unbounded + LCP constraints, and last the purely
  // LCP constraints.
  //
  // joints with m=0 are inactive and are removed from the joints array
  // entirely, so that the code that follows does not consider them.
  int m = 0;
  ALLOCA(dxJoint::Info1,info,nj*sizeof(dxJoint::Info1));
  ALLOCA(int,ofs,nj*sizeof(int));

  for (i=0, j=0; j<nj; j++) {	// i=dest, j=src
    joint[j]->getInfo1 (info+i);
    dIASSERT (info[i].m >= 0 && info[i].m <= 6 &&
	      info[i].nub >= 0 && info[i].nub <= info[i].m);
    if (info[i].m > 0) {
      joint[i] = joint[j];
      i++;
    }
  }
  nj = i;

  // the purely unbounded constraints
  for (i=0; i<nj; i++) if (info[i].nub == info[i].m) {
    ofs[i] = m;
    m += info[i].m;
  }
  //int nub = m;
  // the mixed unbounded + LCP constraints
  for (i=0; i<nj; i++) if (info[i].nub > 0 && info[i].nub < info[i].m) {
    ofs[i] = m;
    m += info[i].m;
  }
  // the purely LCP constraints
  for (i=0; i<nj; i++) if (info[i].nub == 0) {
    ofs[i] = m;
    m += info[i].m;
  }

  // create (6*nb,6*nb) inverse mass matrix `invM', and fill it with mass
  // parameters
#ifdef TIMING
  dTimerNow ("create mass matrix");
#endif
  ALLOCA(dReal, invM, n6*n6);

  dSetZero (invM,n6*n6);
  for (i=0; i<nb; i++) {
    dReal *MM = invM+(i*6)*n6+(i*6);
    MM[0] = body[i]->invMass;
    MM[n6+1] = body[i]->invMass;
    MM[2*n6+2] = body[i]->invMass;
    MM += 3*n6+3;
    for (j=0; j<3; j++) for (k=0; k<3; k++) {
      MM[j*n6+k] = invI[i*12+j*4+k];
    }
  }

  // copy invM to device
  dev_invM = cuda_copyToDevice(invM, n6*n6*sizeof(dReal));

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

  // if there are constraints, compute cforce
  if (m > 0) {
    // create a constraint equation right hand side vector `c', a constraint
    // force mixing vector `cfm', and LCP low and high bound vectors, and an
    // 'findex' vector.
    ALLOCA(dReal,c,m*sizeof(dReal));
    ALLOCA(dReal,cfm,m*sizeof(dReal));
    ALLOCA(dReal,lo,m*sizeof(dReal));
    ALLOCA(dReal,hi,m*sizeof(dReal));
    ALLOCA(int,findex,m*sizeof(int));
    dSetZero (c,m);
    dSetValue (cfm,m,world->global_cfm);
    dSetValue (lo,m,-dInfinity);
    dSetValue (hi,m, dInfinity);
    for (i=0; i<m; i++) findex[i] = -1;

    // create (m,6*nb) jacobian mass matrix `J', and fill it with constraint
    // data. also fill the c vector.
#   ifdef TIMING
    dTimerNow ("create J");
#   endif
    ALLOCA(dReal,J,m*n6*sizeof(dReal));
    dSetZero (J,m*n6);
    dxJoint::Info2 Jinfo;
    Jinfo.rowskip = n6;
    Jinfo.fps = dRecip(stepsize);
    Jinfo.erp = world->global_erp;
    for (i=0; i<nj; i++) {
      Jinfo.J1l = J + n6*ofs[i] + 6*joint[i]->node[0].body->tag;
      Jinfo.J1a = Jinfo.J1l + 3;
      if (joint[i]->node[1].body) {
	Jinfo.J2l = J + n6*ofs[i] + 6*joint[i]->node[1].body->tag;
	Jinfo.J2a = Jinfo.J2l + 3;
      }
      else {
	Jinfo.J2l = 0;
	Jinfo.J2a = 0;
      }
      Jinfo.c = c + ofs[i];
      Jinfo.cfm = cfm + ofs[i];
      Jinfo.lo = lo + ofs[i];
      Jinfo.hi = hi + ofs[i];
      Jinfo.findex = findex + ofs[i];
      joint[i]->getInfo2 (&Jinfo);
      // adjust returned findex values for global index numbering
      for (j=0; j<info[i].m; j++) {
	if (findex[ofs[i] + j] >= 0) findex[ofs[i] + j] += ofs[i];
      }
    }

    // copy J to device
	dev_J = cuda_copyToDevice(J, m*n6*sizeof(dReal));
	cuda_malloc((void **) dev_JinvM, m*n6*sizeof(dReal));

    // compute A = J*invM*J'
#   ifdef TIMING
    dTimerNow ("compute A");
#   endif
    ALLOCA(dReal,JinvM,m*n6*sizeof(dReal));
    //dSetZero (JinvM,m*n6);
    //dMultiply0 (JinvM,J,invM,m,n6,n6);

    cuda_dMultiply0(dev_JinvM, dev_J, dev_invM, m, n6, n6);
    cuda_copyFromDevice(JinvM, dev_JinvM, m*n6*sizeof(dReal));

    int mskip = dPAD(m);
    ALLOCA(dReal,A,m*mskip*sizeof(dReal));
	cuda_malloc((void **) dev_A, m*mskip*sizeof(dReal));

    //dSetZero (A,m*m);
    //dMultiply2 (A,JinvM,J,m,n6,m);

    cuda_dMultiply2(dev_A, dev_JinvM, dev_J, m, n6, m);
    cuda_copyFromDevice(A, dev_A, m*m*sizeof(dReal));

    // add cfm to the diagonal of A
    for (i=0; i<m; i++) A[i*m+i] += cfm[i] * Jinfo.fps;

#   ifdef COMPARE_METHODS
    comparator.nextMatrix (A,m,m,1,"A");
#   endif

    // compute `rhs', the right hand side of the equation J*a=c
#   ifdef TIMING
    dTimerNow ("compute rhs");
#   endif
    ALLOCA(dReal,tmp1,n6*sizeof(dReal));
    //dSetZero (tmp1,n6);
    dMultiply0 (tmp1,invM,fe,n6,n6,1);
    for (i=0; i<n6; i++) tmp1[i] += v[i]/stepsize;
    ALLOCA(dReal,rhs,m*sizeof(dReal));
    //dSetZero (rhs,m);
    dMultiply0 (rhs,J,tmp1,m,n6,1);
    for (i=0; i<m; i++) rhs[i] = c[i]/stepsize - rhs[i];

#   ifdef COMPARE_METHODS
    comparator.nextMatrix (c,m,1,0,"c");
    comparator.nextMatrix (rhs,m,1,0,"rhs");
#   endif



 

#ifndef DIRECT_CHOLESKY
    // solve the LCP problem and get lambda.
    // this will destroy A but that's okay
#   ifdef TIMING
    dTimerNow ("solving LCP problem");
#   endif
    ALLOCA(dReal,lambda,m*sizeof(dReal));
    ALLOCA(dReal,residual,m*sizeof(dReal));
    dSolveLCP (m,A,lambda,rhs,residual,nub,lo,hi,findex);

#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (dMemoryFlag == d_MEMORY_OUT_OF_MEMORY)
      return;
#endif


#else

    // OLD WAY - direct factor and solve

    // factorize A (L*L'=A)
#   ifdef TIMING
    dTimerNow ("factorize A");
#   endif
    ALLOCA(dReal,L,m*m*sizeof(dReal));
    memcpy (L,A,m*m*sizeof(dReal));
    if (dFactorCholesky (L,m)==0) dDebug (0,"A is not positive definite");

    // compute lambda
#   ifdef TIMING
    dTimerNow ("compute lambda");
#   endif
    ALLOCA(dReal,lambda,m*sizeof(dReal));
    memcpy (lambda,rhs,m * sizeof(dReal));
    dSolveCholesky (L,lambda,m);
#endif

#   ifdef COMPARE_METHODS
    comparator.nextMatrix (lambda,m,1,0,"lambda");
#   endif

    // compute the velocity update `vnew'
#   ifdef TIMING
    dTimerNow ("compute velocity update");
#   endif
    dMultiply1 (tmp1,J,lambda,n6,m,1);
    for (i=0; i<n6; i++) tmp1[i] += fe[i];
    dMultiply0 (vnew,invM,tmp1,n6,n6,1);
    for (i=0; i<n6; i++) vnew[i] = v[i] + stepsize*vnew[i];

#ifdef REPORT_ERROR
    // see if the constraint has worked: compute J*vnew and make sure it equals
    // `c' (to within a certain tolerance).
#   ifdef TIMING
    dTimerNow ("verify constraint equation");
#   endif
    dMultiply0 (tmp1,J,vnew,m,n6,1);
    dReal err = 0;
    for (i=0; i<m; i++) {
		err += dFabs(tmp1[i]-c[i]);
    }
	printf ("total constraint error=%.6e\n",err);
#endif

	cuda_free2(dev_J);
	cuda_free2(dev_JinvM);
	cuda_free2(dev_A);
  }
  else {
    // no constraints
    dMultiply0 (vnew,invM,fe,n6,n6,1);
    for (i=0; i<n6; i++) vnew[i] = v[i] + stepsize*vnew[i];
  }
  cuda_free2(dev_invM);

#ifdef COMPARE_METHODS
  comparator.nextMatrix (vnew,n6,1,0,"vnew");
#endif

  // apply the velocity update to the bodies
#ifdef TIMING
  dTimerNow ("update velocity");
#endif
  for (i=0; i<nb; i++) {
    for (j=0; j<3; j++) body[i]->lvel[j] = vnew[i*6+j];
    for (j=0; j<3; j++) body[i]->avel[j] = vnew[i*6+3+j];
  }

  // update the position and orientation from the new linear/angular velocity
  // (over the given timestep)
#ifdef TIMING
  dTimerNow ("update position");
#endif
  for (i=0; i<nb; i++) dxStepBody (body[i],stepsize);

#ifdef TIMING
  dTimerNow ("tidy up");
#endif

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

#ifdef TIMING
  dTimerEnd();
  if (m > 0) dTimerReport (stdout,1);
#endif

}

