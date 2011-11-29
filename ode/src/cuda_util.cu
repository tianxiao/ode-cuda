#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <ode/common.h>
#include <ode/cuda_helper.h>
#include <ode/cuda_matrix.h>
#include "util.h"
#include "cuda_util.h"

#define BLOCKSIZE 4

struct cuda_Island {
	dxBody **body;
	dxJoint **joint;
	int nb;
	int nj;
}

sizeof(cuda_Island) * world->nb

template <int BLOCK_SIZE> __global__ void process_islands()
{
  dxBody *b,*bb,**body;
  dxJoint *j,**joint;

  // nothing to do if no bodies
  if (world->nb <= 0) return;

  // handle auto-disabling of bodies
  dInternalHandleAutoDisabling (world,stepsize);

  // make arrays for body and joint lists (for a single island) to go into
  body = (dxBody**) ALLOCA (world->nb * sizeof(dxBody*));
  joint = (dxJoint**) ALLOCA (world->nj * sizeof(dxJoint*));
  int bcount = 0;	// number of bodies in `body'
  int jcount = 0;	// number of joints in `joint'

  // set all body/joint tags to 0
  for (b=world->firstbody; b; b=(dxBody*)b->next) b->tag = 0;
  for (j=world->firstjoint; j; j=(dxJoint*)j->next) j->tag = 0;

  // allocate a stack of unvisited bodies in the island. the maximum size of
  // the stack can be the lesser of the number of bodies or joints, because
  // new bodies are only ever added to the stack by going through untagged
  // joints. all the bodies in the stack must be tagged!
  int stackalloc = (world->nj < world->nb) ? world->nj : world->nb;
  dxBody **stack = (dxBody**) ALLOCA (stackalloc * sizeof(dxBody*));

  islands = ALLOCA (world->nb * sizeof(cuda_Island));

  for (bb=world->firstbody; bb; bb=(dxBody*)bb->next) {
    // get bb = the next enabled, untagged body, and tag it
    if (bb->tag || (bb->flags & dxBodyDisabled)) continue;
    bb->tag = 1;

    // tag all bodies and joints starting from bb.
    int stacksize = 0;
    b = bb;
    body[0] = bb;
    bcount = 1;
    jcount = 0;
    goto quickstart;
    while (stacksize > 0) {
      b = stack[--stacksize];	// pop body off stack
      body[bcount++] = b;	// put body on body list
      quickstart:

      // traverse and tag all body's joints, add untagged connected bodies
      // to stack
      for (dxJointNode *n=b->firstjoint; n; n=n->next) {
        if (!n->joint->tag && n->joint->isEnabled()) {
	  n->joint->tag = 1;
	  joint[jcount++] = n->joint;
	  if (n->body && !n->body->tag) {
	    n->body->tag = 1;
	    stack[stacksize++] = n->body;
	  }
	}
      }
      dIASSERT(stacksize <= world->nb);
      dIASSERT(stacksize <= world->nj);
    }

	islands[memcpy
	islands[island_count++] = 
  }

    // now do something with body and joint lists
	//stepper (world,island->body,island-nb,island->joint,island->nj,stepsize);
    stepper (world,body,bcount,joint,jcount,stepsize);

    // what we've just done may have altered the body/joint tag values.
    // we must make sure that these tags are nonzero.
    // also make sure all bodies are in the enabled state.
    int i;
    for (i=0; i<bcount; i++) {
      body[i]->tag = 1;
      body[i]->flags &= ~dxBodyDisabled;
    }
    for (i=0; i<jcount; i++) joint[i]->tag = 1;
  }
}

ODE_API void cuda_dxProcessIslands(dxWorld *world, dReal stepsize, dstepper_fn_t stepper)
{
	const int block_size = BLOCKSIZE;
	dim3 dimBlock(block_size, block_size);
	dim3 dimGrid(block_size, block_size);
	process_islands<block_size><<<dimGrid, dimBlock>>>();
}

