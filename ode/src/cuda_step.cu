// cuda_step.cu

//#include "objects.h"
//#include "joints/joint.h"
//#include <ode/odeconfig.h>
//#include "config.h"
//#include <ode/odemath.h>
//#include <ode/rotation.h>
//#include <ode/timer.h>
//#include <ode/error.h>
//#include <ode/matrix.h>
//#include "lcp.h"
#include "util.h"

#include <ode/cuda_step.h>
#include <cuda.h>
#include <ode/cuda_helper.h>
#include <ode/cuda_matrix.h>
#include "cuprintf.cu"

#define BLOCKSIZE 16

__device__ void dQMultiply0 (dQuaternion qa, const dQuaternion qb, const dQuaternion qc) {
  qa[0] = qb[0]*qc[0] - qb[1]*qc[1] - qb[2]*qc[2] - qb[3]*qc[3];
  qa[1] = qb[0]*qc[1] + qb[1]*qc[0] + qb[2]*qc[3] - qb[3]*qc[2];
  qa[2] = qb[0]*qc[2] + qb[2]*qc[0] + qb[3]*qc[1] - qb[1]*qc[3];
  qa[3] = qb[0]*qc[3] + qb[3]*qc[0] + qb[1]*qc[2] - qb[2]*qc[1];
}

__device__ void dDQfromW (dReal dq[4], const dVector3 w, const dQuaternion q)
{
  dq[0] = REAL(0.5)*(- w[0]*q[1] - w[1]*q[2] - w[2]*q[3]);
  dq[1] = REAL(0.5)*(  w[0]*q[0] + w[1]*q[3] - w[2]*q[2]);
  dq[2] = REAL(0.5)*(- w[0]*q[3] + w[1]*q[0] + w[2]*q[1]);
  dq[3] = REAL(0.5)*(  w[0]*q[2] - w[1]*q[1] + w[2]*q[0]);
}

__device__ void dWtoDQ(const dVector3 w, const dQuaternion q, dReal dq[4]) {
	return dDQfromW(dq,w,q);
}

__device__ void dRfromQ (dMatrix3 R, const dQuaternion q)
{
  // q = (s,vx,vy,vz)
  dReal qq1 = 2*q[1]*q[1];
  dReal qq2 = 2*q[2]*q[2];
  dReal qq3 = 2*q[3]*q[3];
  R[(0)*4+(0)] = 1 - qq2 - qq3;
  R[(0)*4+(1)] = 2*(q[1]*q[2] - q[0]*q[3]);
  R[(0)*4+(2)] = 2*(q[1]*q[3] + q[0]*q[2]);
  R[(0)*4+(3)] = (0.0);
  R[(1)*4+(0)] = 2*(q[1]*q[2] + q[0]*q[3]);
  R[(1)*4+(1)] = 1 - qq1 - qq3;
  R[(1)*4+(2)] = 2*(q[2]*q[3] - q[0]*q[1]);
  R[(1)*4+(3)] = (0.0);
  R[(2)*4+(0)] = 2*(q[1]*q[3] - q[0]*q[2]);
  R[(2)*4+(1)] = 2*(q[2]*q[3] + q[0]*q[1]);
  R[(2)*4+(2)] = 1 - qq1 - qq2;
  R[(2)*4+(3)] = (0.0);
}

__device__ void dQtoR(const dQuaternion q, dMatrix3 R) {
	return dRfromQ(R, q);
}

__device__ dReal dDOTpq(dReal *a, dReal *b, int p, int q) {
	return ((a)[0]*(b)[0] + (a)[p]*(b)[q] + (a)[2*(p)]*(b)[2*(q)]);
}

__device__ dReal dDOT(dReal *a, dReal *b) {
	return dDOTpq(a,b,1,1);
}

__device__ dReal dDOT13(dReal *a, dReal *b) {
	return dDOTpq(a,b,1,3);
}

__device__ int dNormalize4(dVector4 a) {
  dReal l = dDOT(a,a)+a[3]*a[3];
  if (l > 0) {
    //l = dRecipSqrt(l);
	l = ((1.0f/sqrtf(l)));
    a[0] *= l;
    a[1] *= l;
    a[2] *= l;
    a[3] *= l;
	return 1;
  }
  else {
    a[0] = 1;
    a[1] = 0;
    a[2] = 0;
    a[3] = 0;
    return 0;
  }
}

/*ODE_API void cuda_dxProcessIslands (dxWorld *world, dReal stepsize, dstepper_fn_t cuda_stepper)
{
	int cuda_bodies_count = 0;
	dxBody *cuda_bodies;
	cudaMalloc((void**) &cuda_bodies, sizeof(dxBody)*world->nb);
	dxBody *bb;
	for (bb=world->firstbody;bb;bb=(dxBody*)bb->next)
		cudaMemcpy(cuda_bodies+sizeof(dxBody)*cuda_bodies_count++, bb, sizeof(dxBody), cudaMemcpyHostToDevice);
    cuda_stepper (world,cuda_bodies,cuda_bodies_count,NULL,NULL,stepsize);*/

/* special-case matrix multiplication functions */

// A = B*C  A, B, C all 3x3
__device__ void cuda_dMultiply0_333(dReal *A, dReal *B, dReal *C) {
	A[0] = dDOT13((B),(C)); 
	A[1] = dDOT13((B),(C+1)); 
	A[2] = dDOT13((B),(C+2)); 
	A[4] = dDOT13((B+4),(C)); 
	A[5] = dDOT13((B+4),(C+1)); 
	A[6] = dDOT13((B+4),(C+2));
	A[8] = dDOT13((B+8),(C)); 
	A[9] = dDOT13((B+8),(C+1)); 
	A[10] = dDOT13((B+8),(C+2)); 
}

// A = B*C^T  A, B, C all 3x3
__device__ void cuda_dMultiply2_333(dReal *A, dReal *B, dReal *C) {
	A[0] = dDOT((B),(C)); 
	A[1] = dDOT((B),(C+4)); 
	A[2] = dDOT((B),(C+8)); 
	A[4] = dDOT((B+4),(C)); 
	A[5] = dDOT((B+4),(C+4)); 
	A[6] = dDOT((B+4),(C+8));
	A[8] = dDOT((B+8),(C)); 
	A[9] = dDOT((B+8),(C+4)); 
	A[10] = dDOT((B+8),(C+8)); 
}

// A = B*C  A 3x1, B 3x3, C 3x1
__device__ void cuda_dMultiply0_331(dReal *A, dReal *B, dReal *C) {
	A[0] = dDOT((B),(C));
	A[1] = dDOT((B+4),(C));
	A[2] = dDOT((B+8),(C));
}

// A = B*C  A 1x3, B 1x3, C 3x3
__device__ void cuda_dMultiply0_133(dReal *A, dReal *B, dReal *C) {
	A[0] = dDOT13((B),(C));
	A[1] = dDOT13((B),(C+1));
	A[2] = dDOT13((B),(C+2));
}

// A += B*C  A 3x1, B 3x3, C 3x1
__device__ void cuda_dMultiplyAdd0_331(dReal *A, dReal *B, dReal *C) {
	A[0] += dDOT((B),(C));
	A[1] += dDOT((B+4),(C));
	A[2] += dDOT((B+8),(C));
}

// a -= b cross c
__device__ void cuda_dCross(dReal *a, dReal *b, dReal *c) {
	a[0] -= ((b)[1]*(c)[2] - (b)[2]*(c)[1]);
	a[1] -= ((b)[2]*(c)[0] - (b)[0]*(c)[2]);
	a[2] -= ((b)[0]*(c)[1] - (b)[1]*(c)[0]);
}

// A = B*C  A pxr, B pxq, C qxr
__device__ void naiveMatMultiply(dReal *A, dReal *B, dReal *C, int p, int q, int r) {
	int i, j, k;
	for (i = 0; i < p; i++) {
		for (j = 0; j < r; j++) {
			for (k = 0; k < q; k++) {
				A[i*r + j] += (B[i*q + k])*(C[k*r + j]);
			}
		}
	}
}

__device__ dReal cuda_sinc(dReal x)
{
	// if |x| < 1e-4 then use a taylor series expansion. this two term expansion
	// is actually accurate to one LS bit within this range if double precision
	// is being used - so don't worry!
	if (fabs(x) < 1.0e-4) return (1.0) - x*x*(0.166666666666666666667);
	else return sinf(x)/x;
}

//****************************************************************************
// the slow, but sure way
// note that this does not do any joint feedback!

// given lists of bodies and joints that form an island, perform a first
// order timestep.
//
// `body' is the body array, `nb' is the size of the array.
// `_joint' is the body array, `nj' is the size of the array.

 __global__ void cuda_step(dxBody *body, int nb, dReal stepsize, dReal g1, dReal g2, dReal g3)
{
	dVector3 gravity; 
	gravity[0] = g1;
	gravity[1] = g2;
	gravity[2] = g3;
	int i,j,k;

	dReal I[3*3], invI[3*3];

	int bid = threadIdx.x + blockDim.x * blockIdx.x;
	if (bid >= nb) { return; }
	cuPrintf("%f\t%f\t%f\n", body[bid].posr.pos[0], body[bid].posr.pos[0], body[bid].posr.pos[0]);

	// for all bodies, compute the inertia tensor and its inverse in the global
	// frame, and compute the rotational force and add it to the torque
	// accumulator.
	// @@@ check computation of rotational force.

	//dSetZero (I,3*nb*4);
	//dSetZero (invI,3*nb*4);
	dReal tmp[9];


    // compute inertia tensor in global frame
    cuda_dMultiply2_333(tmp, body[bid].mass.I, body[bid].posr.R);
    cuda_dMultiply0_333(I, body[bid].posr.R, tmp);
    // compute inverse inertia tensor in global frame
    cuda_dMultiply2_333(tmp, body[bid].invI, body[bid].posr.R);
    cuda_dMultiply0_333(invI, body[bid].posr.R, tmp);
    // compute rotational force
    cuda_dMultiply0_331(tmp, I, body[bid].avel);
    cuda_dCross(body[bid].tacc, body[bid].avel, tmp);


	// add the gravity force to all bodies

    if ((body[bid].flags & dxBodyNoGravity)==0) {
		body[bid].facc[0] += body[bid].mass.mass * gravity[0];
		body[bid].facc[1] += body[bid].mass.mass * gravity[1];
		body[bid].facc[2] += body[bid].mass.mass * gravity[2];
    }


	// create (6*nb,6*nb) inverse mass matrix `invM', and fill it with mass
	// parameters  
	dReal invM[6*6];

	for(i = 0; i < 6*6; i++) invM[i] = 0;



    invM[0] = body[bid].invMass;
    invM[6+1] = body[bid].invMass;
    invM[2*6+2] = body[bid].invMass;
    for (j = 3; j < 6; j++) for (k = 3; k < 6; k++) {
			invM[j*6+k] = invI[j*6+k];
		}
  

	// assemble some body vectors: fe = external forces, v = velocities
	dReal fe[6];
	dReal v[6];

	//dSetZero (fe,n6);
	//dSetZero (v,n6);

    for (j = 0; j < 3; j++) fe[j] = body[bid].facc[j];
    for (j = 0; j < 3; j++) fe[3+j] = body[bid].tacc[j];
    for (j = 0; j < 3; j++) v[j] = body[bid].lvel[j];
    for (j = 0; j < 3; j++) v[3+j] = body[bid].avel[j];

	// this will be set to the velocity update
	dReal vnew[6];
	for(i = 0; i < 6; i++) vnew[i] = 0;

	// no constraints
	naiveMatMultiply(vnew, invM, fe, 6, 6, 1);
	for (i = 0; i < 6; i++) vnew[i] = v[i] + stepsize*vnew[i];

	// apply the velocity update to the bodies

    for (j = 0; j < 3; j++) body[bid].lvel[j] = vnew[j];
    for (j = 0; j < 3; j++) body[bid].avel[j] = vnew[3+j];

	// update the position and orientation from the new linear/angular velocity
	// (over the given timestep)
	//dxBody *b = &(body[bid]);
	dxBody b = (body[bid]);
	// cap the angular velocity
	if (b.flags & dxBodyMaxAngularSpeed) {
        const dReal max_ang_speed = b.max_angular_speed;
        const dReal aspeed = dDOT( b.avel, b.avel );
        if (aspeed > max_ang_speed*max_ang_speed) {
			const dReal coef = max_ang_speed/sqrtf(aspeed);
			//dOPEC(b.avel, *=, coef); // multiply vector by scalar coef
			b.avel[0] *= coef;
			b.avel[1] *= coef;
			b.avel[2] *= coef;
        }
	}
	// end of angular velocity cap

	dReal h = stepsize;

	// handle linear velocity
	for (j=0; j<3; j++) b.posr.pos[j] += h * b.lvel[j];

	if (b.flags & dxBodyFlagFiniteRotation) {
		dVector3 irv;	// infitesimal rotation vector
		dQuaternion q;	// quaternion for finite rotation

		if (b.flags & dxBodyFlagFiniteRotationAxis) {
			// split the angular velocity vector into a component along the finite
			// rotation axis, and a component orthogonal to it.
			dVector3 frv;		// finite rotation vector
			dReal k = dDOT (b.finite_rot_axis,b.avel);
			frv[0] = b.finite_rot_axis[0] * k;
			frv[1] = b.finite_rot_axis[1] * k;
			frv[2] = b.finite_rot_axis[2] * k;
			irv[0] = b.avel[0] - frv[0];
			irv[1] = b.avel[1] - frv[1];
			irv[2] = b.avel[2] - frv[2];

			// make a rotation quaternion q that corresponds to frv * h.
			// compare this with the full-finite-rotation case below.
			h *= REAL(0.5);
			dReal theta = k * h;
			q[0] = cosf(theta);
			dReal s = cuda_sinc(theta) * h;
			q[1] = frv[0] * s;
			q[2] = frv[1] * s;
			q[3] = frv[2] * s;
		}
		else {
			// make a rotation quaternion q that corresponds to w * h
			dReal wlen = sqrtf (b.avel[0]*b.avel[0] + b.avel[1]*b.avel[1] +
								b.avel[2]*b.avel[2]);
			h *= REAL(0.5);
			dReal theta = wlen * h;
			q[0] = cosf(theta);
			dReal s = cuda_sinc(theta) * h;
			q[1] = b.avel[0] * s;
			q[2] = b.avel[1] * s;
			q[3] = b.avel[2] * s;
		}

		// do the finite rotation
		dQuaternion q2;
		dQMultiply0 (q2,q,b.q);
		for (j=0; j<4; j++) b.q[j] = q2[j];

		// do the infitesimal rotation if required
		if (b.flags & dxBodyFlagFiniteRotationAxis) {
			dReal dq[4];
			dWtoDQ (irv,b.q,dq);
			for (j=0; j<4; j++) b.q[j] += h * dq[j];
		}
	}
	else {
		// the normal way - do an infitesimal rotation
		dReal dq[4];
		dWtoDQ (b.avel,b.q,dq);
		for (j=0; j<4; j++) b.q[j] += h * dq[j];
	}

	// normalize the quaternion and convert it to a rotation matrix
	dNormalize4 (b.q);
	dQtoR (b.q,b.posr.R);

	// damping
	if (b.flags & dxBodyLinearDamping) {
		const dReal lin_threshold = b.dampingp.linear_threshold;
        const dReal lin_speed = dDOT( b.lvel, b.lvel );
        if ( lin_speed > lin_threshold) {
			const dReal k = 1 - b.dampingp.linear_scale;
			//dOPEC(b.lvel, *=, k);
			b.lvel[0] *= k;
			b.lvel[1] *= k;
			b.lvel[2] *= k;
        }
	}
	if (b.flags & dxBodyAngularDamping) {
		const dReal ang_threshold = b.dampingp.angular_threshold;
        const dReal ang_speed = dDOT( b.avel, b.avel );
        if ( ang_speed > ang_threshold) {
			const dReal k = 1 - b.dampingp.angular_scale;
			//dOPEC(b.avel, *=, k);
			b.avel[0] *= k;
			b.avel[1] *= k;
			b.avel[2] *= k;
        }
	}


	// zero all force accumulators
    body[bid].facc[0] = 0;
    body[bid].facc[1] = 0;
    body[bid].facc[2] = 0;
    body[bid].facc[3] = 0;
    body[bid].tacc[0] = 0;
    body[bid].tacc[1] = 0;
    body[bid].tacc[2] = 0;
    body[bid].tacc[3] = 0;
}

ODE_API void cuda_dInternalStepIsland_x1 (dxWorld *world, dxBody *cuda_body, int nb, dxJoint * *_joint, int nj, dReal stepsize)
{
	cudaPrintfInit();

	cuda_step<<<nb, 1>>>(cuda_body, world->nb, stepsize, world->gravity[0], world->gravity[1], world->gravity[2]);

	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	//cuda_step<<<BLOCKSIZE/nb, 256>>>(cuda_body, world->nb, stepsize, world->gravity[0], world->gravity[1], world->gravity[2]);
}

 ODE_API void cuda_dxProcessIslands(dxWorld *world, dxBody *cuda_body, dReal stepsize, dstepper_fn_t stepper)
{
	const int block_size = BLOCKSIZE;
	dim3 dimBlock(block_size, block_size);
	dim3 dimGrid(block_size, block_size);

	cuda_dInternalStepIsland_x1(world, cuda_body, world->nb, NULL, 0, stepsize);
}

