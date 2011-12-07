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

// test the step function by comparing the output of the fast and the slow
// version, for various systems. currently you have to define COMPARE_METHODS
// in step.cpp for this to work properly.
//
// @@@ report MAX error

#include <time.h>
#include <ode/ode.h>
#include <ode/common.h>
#include <ode/objects.h>
#include <drawstuff/drawstuff.h>
#include "texturepath.h"

#include <cuda.h>
#include <ode/cuda_helper.h>

#ifdef _MSC_VER
#pragma warning(disable:4244 4305)  // for VC++, no precision loss complaints
#endif

// select correct drawing functions

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#endif


// some constants

#define NUM 5			// number of bodies
#define NUMJ 9			// number of joints
#define SIDE (2)		// side length of a box
#define MASS (1.0)		// mass of a box
#define RADIUS (0.0866)	// sphere radius

static bool gfx = true;
//static bool use_cuda = false;
static bool use_cuda = true;

// dynamics and collision objects

static dWorldID world=0;
static dBodyID body[NUM];
static dJointID joint[NUMJ];

static dBodyID cuda_body;
static dBodyID b_buff;

// create the test system

void createTest()
{
  int i,j;
  if (world) dWorldDestroy (world);

  world = dWorldCreate();

  const dReal scale = 0.1;
  const dReal fmass = 0.1;

  // create random bodies
  for (i=0; i<NUM; i++) {
    // create bodies at random position and orientation
    body[i] = dBodyCreate (world);
    dBodySetPosition (body[i],(dRandReal()*2-1),(dRandReal()*2-1),
		      (dRandReal()*2+RADIUS));
    dReal q[4];
    for (j=0; j<4; j++) q[j] = (dRandReal()*2-1);
    dBodySetQuaternion (body[i],q);

    // set random velocity
    dBodySetLinearVel (body[i], scale*(dRandReal()*2-1),scale*(dRandReal()*2-1),
		       scale*(dRandReal()*2-1));
    dBodySetAngularVel (body[i], scale*(dRandReal()*2-1),scale*(dRandReal()*2-1),
			scale*(dRandReal()*2-1));

    // set random mass (random diagonal mass rotated by a random amount)
    dMass m;
    dMatrix3 R;
    dMassSetBox (&m,1,(fmass+0.1),(fmass+0.1),(fmass+0.1));
    dMassAdjust (&m,(fmass+1));
    for (j=0; j<4; j++) q[j] = (fmass*2-1);
    dQtoR (q,R);
    dMassRotate (&m,R);
    dBodySetMass (body[i],&m);
  }
}

void cuda_createTest()
{
  int i,j;
  if (world) dWorldDestroy(world);
  world = dWorldCreate();
  const dReal scale = 0.1;
  const dReal fmass = 0.1;
  // create random bodies
  for (i=0; i<NUM; i++) {
    // create bodies at random position and orientation
    body[i] = dBodyCreate (world);
    dBodySetPosition (body[i],(dRandReal()*2-1),(dRandReal()*2-1),
		      (dRandReal()*2+RADIUS));
    dReal q[4];
    for (j=0; j<4; j++) q[j] = (dRandReal()*2-1);
    dBodySetQuaternion (body[i],q);

    // set random velocity
    dBodySetLinearVel (body[i], scale*(dRandReal()*2-1),scale*(dRandReal()*2-1),
		       scale*(dRandReal()*2-1));
    dBodySetAngularVel (body[i], scale*(dRandReal()*2-1),scale*(dRandReal()*2-1),
			scale*(dRandReal()*2-1));

    // set random mass (random diagonal mass rotated by a random amount)
    dMass m;
    dMatrix3 R;
    dMassSetBox (&m,1,(fmass+0.1),(fmass+0.1),(fmass+0.1));
    dMassAdjust (&m,(fmass+1));
    for (j=0; j<4; j++) q[j] = (fmass*2-1);
    dQtoR (q,R);
    dMassRotate (&m,R);
    dBodySetMass (body[i],&m);
  }
  cuda_copyBodiesToDevice(cuda_body, body, NUM);
}

// start simulation - set viewpoint

static void start()
{
  dAllocateODEDataForThread(dAllocateMaskAll);

  if (gfx) {
    static float xyz[3] = {2.6117f,-1.4433f,2.3700f};
    static float hpr[3] = {151.5000f,-30.5000f,0.0000f};
    dsSetViewpoint (xyz,hpr);
  }
  createTest();
}

static void cuda_start()
{
	b_buff = (dBodyID) malloc(sizeof(body[0])*NUM);
  dAllocateODEDataForThread(dAllocateMaskAll);
  if (gfx) {
    static float xyz[3] = {2.6117f,-1.4433f,2.3700f};
    static float hpr[3] = {151.5000f,-30.5000f,0.0000f};
    dsSetViewpoint (xyz,hpr);
  }
  cuda_createTest();
}

// simulation loop

static void simLoop (int pause)
{
  if (!pause) {
    // add random forces and torques to all bodies
    int i;
    const dReal scale1 = 0.005;
    const dReal scale2 = 0.005;
    for (i=0; i<NUM; i++) {
      dBodyAddForce (body[i],
		     scale1*(dRandReal()*2-1),
		     scale1*(dRandReal()*2-1),
		     scale1*(dRandReal()*2-1));
      dBodyAddTorque (body[i],
		     scale2*(dRandReal()*2-1),
		     scale2*(dRandReal()*2-1),
		     scale2*(dRandReal()*2-1));
    }

    dWorldStep (world,0.005);
  }

  // float sides[3] = {SIDE,SIDE,SIDE};
  if (gfx) {
    dsSetColor (1,1,0);
    for (int i=0; i<NUM; i++)
      dsDrawSphere (dBodyGetPosition(body[i]), dBodyGetRotation(body[i]),RADIUS);
  }
}

static void cuda_simLoop (int pause)
{
  if (!pause) {
    // add random forces and torques to all bodies
    int i;
    const dReal scale1 = 0.005;
    const dReal scale2 = 0.005;
	for (i=0; i<NUM; i++) {
/*		dBodyAddForce (body[i],
		               scale1*(dRandReal()*2-1),
		               scale1*(dRandReal()*2-1),
		               scale1*(dRandReal()*2-1));
		dBodyAddTorque (body[i],
		                scale2*(dRandReal()*2-1),
		                scale2*(dRandReal()*2-1),
		                scale2*(dRandReal()*2-1));
						*/
    }
    //cuda_dWorldStep (world,0.005);
	//dWorldStep (world,0.005);

	//cuda_copyBodiesToDevice2(cuda_body, world, NUM);
	cuda_dxProcessIslands(world, cuda_body, 0.005, NULL);
	if (gfx) {
		cuda_copyBodiesFromDevice(world, cuda_body, NUM, b_buff);
	}
  }

  // float sides[3] = {SIDE,SIDE,SIDE};
  if (gfx) {
    dsSetColor (1,1,0);
    for (int i=0; i<NUM; i++)
      dsDrawSphere (dBodyGetPosition(body[i]), dBodyGetRotation(body[i]),RADIUS);
  }
}

int main (int argc, char **argv)
{
  if (use_cuda) {
    cuda_body = cuda_initBodiesOnDevice(NUM);
  }

  // setup pointers to drawstuff callback functions
  dsFunctions fn;
  fn.version = DS_VERSION;
  if (use_cuda) {
    fn.start = &cuda_start;
    fn.step = &cuda_simLoop;
  } else {
    fn.start = &start;
    fn.step = &simLoop;
  }
  fn.command = 0;
  fn.stop = 0;
  fn.path_to_textures = DRAWSTUFF_TEXTURE_PATH;

  dInitODE2(0);
  dRandSetSeed (time(0));

  // run simulation
  dsSimulationLoop (argc,argv,352,288,&fn);

  cuda_free(cuda_body);

  dWorldDestroy(world);
  dCloseODE();
  return 0;
}

