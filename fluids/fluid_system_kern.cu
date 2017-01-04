#define CUDA_KERNEL
#include "fluid_system_kern.cuh"
#include "cutil_math.h"
#include "radixsort.cu"						// Build in RadixSort
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

__constant__ fluidParams  simData;
__constant__ sysParams    sysData;
#define EPSILON   0.00001 




__global__ void insertParticles ( fluidBufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	register float3 gridMin   = sysData.gridMin;
	register float3 gridDelta = sysData.gridDelta;
	register int3   gridRes   = sysData.gridRes;
	register int3   gridScan  = sysData.gridScanMax;
	register float  poff = sysData.psmoothradius / sysData.psimscale;

	register float3 gcf = (buf.mpos[i] - gridMin) * gridDelta; 
	register int3 gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	register int gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	
	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) 
	{
		buf.mgcell[i] = gs;											// Grid cell insert.
		buf.mgndx[i] = atomicAdd ( &buf.mgridcnt[ gs ], 1 );		// Grid counts.
		gcf = (-make_float3(poff,poff,poff) + buf.mpos[i] - gridMin) * gridDelta;
		gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;		
	} 
	else 
	{
		buf.mgcell[i] = GRID_UNDEF;		
	}
}




// Counting Sort - Full (deep copy)
__global__ void countingSortFull ( fluidBufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = *(uint*) (buf.msortbuf + pnum*BUF_GCELL + i*sizeof(uint) );
	uint indx =  *(uint*) (buf.msortbuf + pnum*BUF_GNDX + i*sizeof(uint) );		

	if ( icell != GRID_UNDEF ) 
	{	  
		// Determine the sort_ndx, location of the particle after sort
	    int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset		
		// Find the original particle data, offset into unsorted buffer (msortbuf)
		char* bpos = buf.msortbuf + i*sizeof(float3);
		// Transfer data to sort location
		buf.mgrid[ sort_ndx ] =		sort_ndx;			// full sort, grid indexing becomes identity		
		buf.mpos[ sort_ndx ] =		*(float3*) (bpos);
		buf.mvel[ sort_ndx ] =		*(float3*) (bpos+ pnum*BUF_VEL);
		buf.mveleval[ sort_ndx ]  =	*(float3*) (bpos+ pnum*BUF_VELEVAL);
		buf.mclr[ sort_ndx ]      = *(uint*)   (buf.msortbuf + pnum*BUF_CLR    + i*sizeof(uint));		// ((uint) 255)<<24; -- dark matter
		buf.mId[ sort_ndx ]       = *(int*)    (buf.msortbuf + pnum*BUF_ID     + i*sizeof(int));		// ((uint) 255)<<24; -- dark matter
		buf.mIsBoundary [sort_ndx]= *(bool*)   (buf.msortbuf + pnum*BUF_ISBOUNDARY + i*sizeof(bool));	// ((uint) 255)<<24; -- dark matter
		buf.mNorm[sort_ndx]       = *(float3*) (buf.msortbuf + pnum*BUF_NORM + i*sizeof(float3));		// ((uint) 255)<<24; -- dark matter

		buf.mgcell[ sort_ndx ] =	icell;
		buf.mgndx[ sort_ndx ]  =	indx;	

	}
}


__device__ float contributePressure ( int i, float3 p, int cell, fluidBufList buf)
{			
	float sum= 0.0;;
	register float d2 = sysData.d2;
	register float r2 = sysData.rd2;
	
	if ( buf.mgridcnt[cell] == 0 ) {sum = 0.0;}
	else
	{
		int cfirst = buf.mgridoff[ cell ];
		int clast = cfirst + buf.mgridcnt[ cell ];	
		for ( int cndx = cfirst; cndx < clast; cndx++ ) 
		{
			int j = buf.mgrid[ cndx ];	
			float length = 2.0*sysData.radius;
			float3 dist = p-buf.mpos[j]; 			
			float de = sqrt(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);  // calculate arc length 
			float dg = length*asin(de/length);
			float dsq = dg*dg;
			if ( dsq <= r2 && dsq >= 0.0) 
			{
				float c = (r2 - dsq)*d2;
				float c3 = c * c * c;	
				c3*=simData.pmass;
				sum += c3;	
			}
			
		}
	}
	return (sum);
}
	


__global__ void computePressure ( fluidBufList buf, int pnum, float restDen )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (sysData.gridRes.z + 1)*sysData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;


	// Sum Density
	register float3 pos = buf.mpos[ i ];
	float sum = 0.0;
	for (int c=0; c < sysData.gridAdjCnt; c++) 
	{
		sum += contributePressure ( i, pos, gc + sysData.gridAdj[c], buf);
	}
	sum *= sysData.poly6kern;
			
	__syncthreads();


	if ( sum == 0.0 ) { buf.mdensity[ i ]= restDen; buf.mperDensity[ i ]  = 1.0/restDen; buf.mpress[ i ] =0.0; return;  } 
	buf.mdensity[ i ]= sum;
	buf.mpress[ i ] = ( sum - restDen) * sysData.pintstiff;
	buf.mperDensity[i] = 1.0/sum;
}



__global__ void computeNormal( fluidBufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	float3 pos = buf.mpos[i];
	float d = sqrt(pos.x*pos.x+pos.y*pos.y+pos.z*pos.z);
	buf.mNorm[i] = make_float3(pos.x/d,pos.y/d,pos.z/d);
}



__device__ float3 contributeForceFluid ( int i, int cell, fluidBufList buf,float dens)
{	

	//if i is boundary it should only have boundary neighbors
	float3 force = make_float3(0,0,0);  //force from fluid-fluid
	if ( buf.mgridcnt[cell] == 0 ) { force = make_float3(0,0,0);}
	else
	{
		for ( int cndx = buf.mgridoff[ cell ]; cndx < buf.mgridoff[ cell ] + buf.mgridcnt[ cell ]; cndx++ ) 
		{										
			int j = buf.mgrid[ cndx ];				
			float3 dist = buf.mpos[i]-buf.mpos[j]; 
			float length = 2.0*sysData.radius;
			float de = sqrt(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);  // calculate arc length 
			float dg = length*asin(de/length);
			float dsq2 = dg*dg;

		
			float r2 = sysData.rd2;
			float r = sysData.psmoothradius;
			if ( dsq2 < r2 && dsq2 > 0) 
			{	
				float dsq = dg*sysData.psimscale;
				float c = ( r - dsq ); 
				if(dsq2 < r2)
				{
					c = ( r - dsq ); 
					float pterm = -1.0*sysData.psimscale *  c * sysData.spikykern * ( buf.mpress[ i ]* buf.mperDensity[ i ]* buf.mperDensity[ i ] + buf.mpress[ j ]* buf.mperDensity[ j ]* buf.mperDensity[ j ] ) / dsq;
					force += (pterm * dist*c);
				}
			}
		}
	}		
	return (force*simData.pmass);
}




__global__ void computeForce ( fluidBufList buf, int pnum ,float dens)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (sysData.gridRes.z + 1)*sysData.gridRes.x + 1;

	// Sum Pressures	
	register float3 forceF = make_float3(0,0,0);		

	for (int c=0; c < sysData.gridAdjCnt; c++) 
	{
		forceF += contributeForceFluid ( i, gc + sysData.gridAdj[c], buf, dens );
	}
	
	buf.mforce[ i ] = forceF - buf.mNorm[i]*dot(buf.mNorm[i],forceF);    //all the particles are surface particles
}	


__global__ void advanceParticles ( float time, float dt,  fluidBufList buf, int numPnts)
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	
	buf.mPoPos[i] = buf.mpos[i];

	if ( buf.mgcell[i] == GRID_UNDEF ) 
	{
		buf.mPoPos[i] = make_float3(-1000,-1000,-1000);
		buf.mvel[i] = make_float3(0,0,0);
		return;
	}
			
	// Get particle vars
	register float speed;
	register float3 accel;

	// Leapfrog integration						
	accel = buf.mforce[i];

	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > simData.AL2 ) 
	{
		accel *= simData.AL / sqrt(speed);
	}

	// Velocity Limit
	float3 vel = buf.mvel[i];
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > simData.VL2 ) 
	{
		speed = simData.VL2;
		vel *= simData.VL / sqrt(speed);
	}

	float3 vnext    = accel*dt + vel;				// v(t+1/2) = v(t-1/2) + a(t) dt		
	buf.mveleval[i] = (vel + vnext) * 0.5;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	buf.mvel[i]  = vnext;

	float3 step = vnext * (dt/sysData.psimscale);
	float movedist = step.x*step.x+step.y*step.y+step.z*step.z;
		
	if(movedist>EPSILON)
	{
		buf.mPoPos[i] += step;						// p(t+1) = p(t) + v(t+1/2) dt	
	}
	buf.mvel[i] *=0.9;      // behave as viscosity
}


__global__ void updateColor(fluidBufList buf, int numPnts , int c)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	if (c==0)   //four colors
	{
		if(buf.mNorm[i].y>0&&buf.mNorm[i].x>0)
		{
			buf.mclr[ i ] = 0xFF0000FF;
		}
		else if(buf.mNorm[i].y<0&&buf.mNorm[i].x>0)
		{
			buf.mclr[ i ] = 0xFF00FF00;
		}
		else if(buf.mNorm[i].y<0&&buf.mNorm[i].x<0)
		{
			buf.mclr[ i ] = 0xFFFF0000;
		}
		else
		{
			buf.mclr[ i ] = 0xFFFF00FF;
		}
		
	}
	else if(c==1)        // show half sphere
	{
		if(buf.mNorm[i].z>0)
		{
			buf.mclr[ i ] = 0xFFFF0000;
		}
		else
		{
			buf.mclr[ i ] = 0x00FF0000;
		}
	}
	else if(c==2)        // one color
	{
		buf.mclr[ i ] = 0xFFFF0000;
	}
}

__global__ void updateParticles ( fluidBufList buf, int numPnts )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	float3 p = buf.mPoPos[i];
	float l2 = p.x*p.x+p.y*p.y+p.z*p.z;

	if(l2!=sysData.radius*sysData.radius&&l2>0)
	{
		float l=sqrt(l2);
		buf.mpos[i] = make_float3(p.x*sysData.radius/l,p.y*sysData.radius/l,p.z*sysData.radius/l);//projectionµÄ¹ý³Ì¡£
	}
	else
	{
		buf.mvel[i] = make_float3(0,0,0);
	    buf.mveleval[i] = make_float3(0,0,0);
	}
}
	

void updateSysParams(sysParams* sysp)
{
	#ifdef CUDA_42
		cudaMemcpyToSymbol ( "sysData", sysp, sizeof(sysParams) );
	#else
		cudaMemcpyToSymbol ( sysData, sysp, sizeof(sysParams) );
	#endif
}


void updateSimParams ( fluidParams* cpufp)
{
	#ifdef CUDA_42
		cudaMemcpyToSymbol ( "simData", cpufp, sizeof(fluidParams) );
	#else
		cudaMemcpyToSymbol ( simData, cpufp, sizeof(fluidParams) );
	#endif
}









