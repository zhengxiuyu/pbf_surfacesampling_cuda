

#include <conio.h>
#include <cutil_math.h>				// cutil32.lib
#include <string.h>
#include <assert.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN  
#endif
#include <windows.h>
#include "cuda.h"

#include <stdio.h>
#include <math.h>

extern void app_printf ( char* format, ... );
extern void app_printEXIT ( char* format, ... );
extern char app_getch ();

#define		MAX_NBR			80

#include "fluid_system_host.cuh"		
#include "fluid_system_kern.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

fluidParams		fcpu;		// CPU Fluid params
fluidParams*	fcuda;		// GPU Fluid params
fluidBufList	fbuf;		// GPU Particle buffers
sysParams		syscpu;		// CPU obj params


float*  density;
bool cudaCheck ( cudaError_t status, char* msg )
{
	if ( status != cudaSuccess ) 
	{
		app_printf ( "CUDA ERROR: %s\n", cudaGetErrorString ( status ) );
		app_getch ();

		MessageBox ( NULL, cudaGetErrorString ( status), msg, MB_OK );
		return false;
	} 
	return true;
}


void cudaExit ()
{
	cudaDeviceReset();
}

// Initialize CUDA
void cudaInit()
{   
	int count = 0;
	int i = 0;

	cudaError_t err = cudaGetDeviceCount(&count);
	if ( err==cudaErrorInsufficientDriver) { app_printEXIT( "CUDA driver not installed.\n"); }
	if ( err==cudaErrorNoDevice) { app_printEXIT ( "No CUDA device found.\n"); }
	if ( count == 0) { app_printEXIT ( "No CUDA device found.\n"); }

	for(i = 0; i < count; i++) 
	{
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if(prop.major >= 1) break;
		}
	}
	if(i == count) { app_printEXIT ( "No CUDA device found.\n");  }
	cudaSetDevice(i);

	app_printf( "CUDA initialized.\n");
 
	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);
	
	app_printf ( "-- CUDA --\n" );
	app_printf ( "Name:       %s\n",	p.name );
	app_printf ( "Revision:   %d.%d\n", p.major, p.minor );
	app_printf ( "Global Mem: %d\n", p.totalGlobalMem );
	app_printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	app_printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	app_printf ( "Warp Size:  %d\n", p.warpSize );
	app_printf ( "Mem Pitch:  %d\n", p.memPitch );
	app_printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	app_printf ( "Const Mem:  %d\n", p.totalConstMem );
	app_printf ( "Clock Rate: %d\n", p.clockRate );	

	//fbuf.mgridactive = 0x0;
	preallocBlockSumsInt ( 1 );
};
	
// Compute number of blocks to create
int iDivUp (int a, int b) 
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeNumBlocks (int numPnts, int maxThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( maxThreads, numPnts );
    numBlocks = iDivUp ( numPnts, numThreads );
}

void FluidClearCUDA ()
{
	cudaCheck ( cudaFree ( fbuf.mNorm),			"Free mNorm" );     
	cudaCheck ( cudaFree ( fbuf.mPoPos),		"Free mPoPos" );	
	cudaCheck ( cudaFree ( fbuf.mpos ),			"Free mpos" );	
	cudaCheck ( cudaFree ( fbuf.mvel ),			"Free mvel" );	
	cudaCheck ( cudaFree ( fbuf.mveleval ),		"Free mveleval" );	
	cudaCheck ( cudaFree ( fbuf.mforce ),		"Free mforce" );
	cudaCheck ( cudaFree ( fbuf.mpress ),		"Free mpress");	
	cudaCheck ( cudaFree ( fbuf.mdensity ),		"Free mdensity" );	
	cudaCheck ( cudaFree ( fbuf.mperDensity),	"free mperDensity");	
	cudaCheck ( cudaFree ( fbuf.mgcell ),		"Free mgcell" );	
	cudaCheck ( cudaFree ( fbuf.mgndx ),		"Free mgndx" );	
	cudaCheck ( cudaFree ( fbuf.mclr ),			"Free mclr" );	
	cudaCheck ( cudaFree ( fbuf.msortbuf ),		"Free msortbuf" );	
	cudaCheck ( cudaFree ( fbuf.mgrid ),		"Free mgrid" );
	cudaCheck ( cudaFree ( fbuf.mgridcnt ),		"Free mgridcnt" );
	cudaCheck ( cudaFree ( fbuf.mgridoff ),		"Free mgridoff" );
	cudaCheck ( cudaFree ( fbuf.mIsBoundary  ),	"Free misshow" );
	cudaCheck ( cudaFree ( fbuf.mId),           "Free mId" ); 

}


void SetupSysCUDA(int gsrch, int3 res, float3 delta, float3 gmin,  int total)
{
	syscpu.gridRes = res;
	syscpu.gridDelta = delta;
	syscpu.gridMin = gmin;
	syscpu.gridTotal = total;
	syscpu.gridSrch = gsrch; 
	syscpu.gridAdjCnt = gsrch*gsrch*gsrch;
	syscpu.gridScanMax = res;
	syscpu.gridScanMax -= make_int3( syscpu.gridSrch, syscpu.gridSrch, syscpu.gridSrch );
	// Build Adjacency Lookup
	int cell = 0;
	for (int y=0; y < gsrch; y++ )
	{
		for (int z=0; z < gsrch; z++ )
		{
			for (int x=0; x < gsrch; x++ )
			{
				syscpu.gridAdj [ cell++]  = ( y * syscpu.gridRes.z+ z )*syscpu.gridRes.x +  x ;
			}
		}
	}
	int threadsPerBlock = 192;
	computeNumBlocks ( syscpu.gridTotal, threadsPerBlock, syscpu.gridBlocks, syscpu.gridThreads);	
	syscpu.szGrid = (syscpu.gridBlocks * syscpu.gridThreads);
	updateSysParams( &syscpu );

	// Prefix Sum - Preallocate Block sums for Sorting
	deallocBlockSumsInt ();
	preallocBlockSumsInt ( syscpu.gridTotal );
	cudaThreadSynchronize ();
}


void SetupFluidCUDA ( int num )
{	
	// Allocate the sim parameters
	fcpu.pnum = num;

	// Compute number of blocks and threads
	int threadsPerBlock = 192;
    computeNumBlocks ( fcpu.pnum, threadsPerBlock, fcpu.numBlocks, fcpu.numThreads);				// particles
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mPoPos,		num*sizeof(float)*3 ),	"Malloc mPoPos" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mpos,		num*sizeof(float)*3 ),	"Malloc mpos" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mvel,		num*sizeof(float)*3 ),	"Malloc mvel" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mveleval,	num*sizeof(float)*3 ),	"Malloc mveleval" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mforce,		num*sizeof(float)*3 ),	"Malloc mforce" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mNorm,		num*sizeof(float)*3 ),	"Malloc mNorm" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mpress,		num*sizeof(float) ),	"Malloc mpress" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mdensity,	num*sizeof(float) ),	"Malloc mdensity" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mperDensity,num*sizeof(float) ),	"Malloc mperDensity" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mgcell,		num*sizeof(uint) ),		"Malloc mgcell" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mgndx,		num*sizeof(uint)),		"Malloc mgndx" );	
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mclr,		num*sizeof(uint) ),		"Malloc mclr" );			
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mIsBoundary,num*sizeof(bool) ),	"Malloc mIsBoundary" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mId,		num*sizeof(int) ),	"Malloc mpos" );
	//mpos mvel mveleval mNorm; mclr gcell gndx; isboundary
	int temp_size =4*(sizeof(float)*3) + 3*sizeof(uint)+sizeof(bool)+sizeof(int);

	cudaCheck ( cudaMalloc ( (void**) &fbuf.msortbuf,	num*temp_size ),		"Malloc msortbuf" );
	
	density = (float*)	malloc ( num*sizeof(float) );

	// Allocate grid
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mgrid,		 num*sizeof(int) ),					"Malloc mgrid" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mgridcnt,	 syscpu.szGrid*sizeof(int) ),		"Malloc mgridcnt" );
	cudaCheck ( cudaMalloc ( (void**) &fbuf.mgridoff,	 syscpu.szGrid*sizeof(int) ),		"Malloc mgridoff" );
}



void SetParamCUDA (float r, float ss, float sr, float mass, float rest,  float istiff,   float al, float vl)
{

	fcpu.pmass = mass;
	fcpu.prest_dens = rest;	
	fcpu.AL = al;
	fcpu.AL2 = al * al;
	fcpu.VL = vl;
	fcpu.VL2 = vl * vl;

	syscpu.radius = r;;
	syscpu.psimscale = ss;
	syscpu.psmoothradius = sr;
	syscpu.r2 = sr * sr;
	syscpu.pintstiff = istiff;


	//**************************pay attention about the kernel used here is for 2D****************************************
	//syscpu.poly6kern = 256.0f / (64.0f *ss* 3.141592f  *pow( sr, 8.0f) );     
	//syscpu.poly6kernGra = (-768.0f / (32.0f *ss* 3.141592f * pow(sr, 8.0f)));
	//syscpu.poly6kernLap = (-768.0f / (32.0f *ss* 3.141592f * pow(sr, 8.0f)));
	//syscpu.spikykern = -30.0f / (3.141592f * ss*pow( sr, 5.0f) );
	//syscpu.lapkern = 20.0f / (3.141592f *ss* pow( sr, 5.0f) );	
	//syscpu.cohesion = 26.0f/(3.141592f *ss* pow(sr, 8.0f));

	//**************************3D kernel****************************************
	syscpu.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
	syscpu.poly6kernGra = (-945.0f / (32.0f * 3.141592f * pow(sr, 9.0f)));
	syscpu.poly6kernLap = (-945.0f / (32.0f * 3.141592f * pow(sr, 9.0f)));
	syscpu.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );
	syscpu.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );
	syscpu.cohesion = 32.0f/(3.141592f * pow(sr, 9.0f));

	syscpu.d2 = syscpu.psimscale * syscpu.psimscale;
	syscpu.rd2 = syscpu.r2 / syscpu.d2;

	// Transfer sim params to device
	updateSimParams( &fcpu );
	updateSysParams( &syscpu );
	cudaThreadSynchronize ();
	
}


void CopyFluidToCUDA ( float* pos,  char* clr, int* id)
{
	// Send particle buffers
	int numPoints = fcpu.pnum;
	cudaMemset (fbuf.mIsBoundary,     0, numPoints*sizeof(bool));
	cudaMemset (fbuf.mNorm,           0, numPoints*sizeof(float)*3 );	
	cudaMemset (fbuf.mvel,            0, numPoints*sizeof(float)*3 );
	cudaMemset (fbuf.mveleval,        0, numPoints*sizeof(float)*3 );
	cudaMemset (fbuf.mforce,          0, numPoints*sizeof(float)*3 );
	cudaMemset (fbuf.mdensity,        0, numPoints*sizeof(float) );
	cudaMemset (fbuf.mperDensity,     0, numPoints*sizeof(float) );
	cudaMemset (fbuf.mpress,          0, numPoints*sizeof(float) );

	cudaCheck( cudaMemcpy ( fbuf.mpos,		pos,		numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ), 	"Memcpy mpos ToDev" );	
	cudaCheck( cudaMemcpy ( fbuf.mPoPos,    pos,		numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ), 	"Memcpy mpos ToDev" );	
	cudaCheck( cudaMemcpy ( fbuf.mclr,		clr,		numPoints*sizeof(uint),   cudaMemcpyHostToDevice ), 	"Memcpy mclr ToDev"  );
	cudaCheck( cudaMemcpy ( fbuf.mId,       id,         numPoints*sizeof(int) , cudaMemcpyHostToDevice ), 	"Memcpy mMass ToDev" );
	cudaThreadSynchronize ();	
}


void InsertParticlesCUDA ()
{
	cudaMemset ( fbuf.mgridcnt, 0,	syscpu.gridTotal * sizeof(int));
	insertParticles<<< fcpu.numBlocks, fcpu.numThreads>>> ( fbuf, fcpu.pnum );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
	{
		fprintf ( stderr,  "CUDA ERROR: InsertParticlesCUDA: %s\n", cudaGetErrorString(error) );
	}  
	cudaThreadSynchronize ();
}



void PrefixSumCellsCUDA ()
{
	// Prefix Sum - determine grid offsets
    prescanArrayRecursiveInt ( fbuf.mgridoff, fbuf.mgridcnt, syscpu.gridTotal, 0);
	cudaThreadSynchronize ();
}



void CountingSortFullCUDA ()
{
	// Transfer particle data to temp buffers
	int n = fcpu.pnum;
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_POS,			fbuf.mpos,			 n*sizeof(float)*3,		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mpos DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_VEL,			fbuf.mvel,			 n*sizeof(float)*3,		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mvel DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_VELEVAL,		fbuf.mveleval,		 n*sizeof(float)*3,		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mveleval DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_CLR,			fbuf.mclr,	         n*sizeof(uint),		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mclr DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_ID,			fbuf.mId,	         n*sizeof(int),		    cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mclr DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_ISBOUNDARY,	fbuf.mIsBoundary,	 n*sizeof(bool),		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mclr DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_NORM,		fbuf.mNorm,			 n*sizeof(float)*3,	    cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mdens DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_GCELL,		fbuf.mgcell,	     n*sizeof(uint),		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mgcell DevToDev" );
	cudaCheck ( cudaMemcpy ( fbuf.msortbuf + n*BUF_GNDX,		fbuf.mgndx,		     n*sizeof(uint),		cudaMemcpyDeviceToDevice ),		"Memcpy msortbuf->mgndx DevToDev" );
	// Counting Sort - pass one, determine grid counts
	cudaMemset ( fbuf.mgrid,	GRID_UCHAR,	fcpu.pnum * sizeof(uint) );

	countingSortFull <<< fcpu.numBlocks, fcpu.numThreads>>> ( fbuf, fcpu.pnum );		
	cudaThreadSynchronize ();
}



void ComputePressureCUDA ()
{

	computePressure<<< fcpu.numBlocks, fcpu.numThreads>>> ( fbuf, fcpu.pnum, fcpu.prest_dens);	
	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
	{
		fprintf ( stderr, "CUDA ERROR: ComputePressureCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaCheck( cudaMemcpy ( density, fbuf.mdensity,	fcpu.pnum*sizeof(float),  cudaMemcpyDeviceToHost ), 	"Memcpy mObjPos FromDev"  );
	cudaThreadSynchronize ();
}


void ComputeNormalCUDA()
{
	computeNormal<<< fcpu.numBlocks, fcpu.numThreads>>> ( fbuf, fcpu.pnum);
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
	{
		fprintf ( stderr,  "CUDA ERROR: ComputeForceCUDA: %s\n", cudaGetErrorString(error) );
	}
	cudaThreadSynchronize ();
}


void ComputeForceCUDA ()
{
	computeForce<<< fcpu.numBlocks, fcpu.numThreads>>> ( fbuf, fcpu.pnum,fcpu.prest_dens);
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
	{
		fprintf ( stderr,  "CUDA ERROR: ComputeForceCUDA: %s\n", cudaGetErrorString(error) );
	}
	cudaThreadSynchronize ();
}

void AdvanceCUDA ( float tm, float dt)
{
	advanceParticles<<< fcpu.numBlocks, fcpu.numThreads>>> ( tm, dt, fbuf, fcpu.pnum);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
	{
		fprintf ( stderr,  "CUDA ERROR: AdvanceCUDA: %s\n", cudaGetErrorString(error) );
	}    
    cudaThreadSynchronize ();
}




void finalize ()
{
	updateParticles<<< fcpu.numBlocks, fcpu.numThreads>>> (  fbuf, fcpu.pnum);
	cudaThreadSynchronize ();
}


void Changecolor(int c)
{
	updateColor<<< fcpu.numBlocks, fcpu.numThreads>>> (  fbuf, fcpu.pnum,c);
	cudaThreadSynchronize ();
}

void CopyFluidFromCUDA( float* pos, float* norm, char* clr)
{
	int numPoints = fcpu.pnum;
	if ( pos != 0x0 ) cudaCheck( cudaMemcpy ( pos,	fbuf.mpos,	numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ),	"Memcpy mpos FromDev"  );
	if ( norm!= 0x0 ) cudaCheck( cudaMemcpy ( norm,	fbuf.mNorm,	numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ),	"Memcpy mpos FromDev"  );
	if ( clr != 0x0 ) cudaCheck( cudaMemcpy ( clr,	fbuf.mclr,		fcpu.pnum*sizeof(uint),  cudaMemcpyDeviceToHost ), 		"Memcpy mclr FromDev"  );
	cudaThreadSynchronize ();
}


// includes, kernels
#include <assert.h>

inline bool isPowerOfTwo(int n) { return ((n&(n-1))==0) ; }

inline int floorPow2(int n) 
{
	#ifdef WIN32
		return 1 << (int)logb((float)n);
	#else
		int exp;
		frexp((float)n, &exp);
		return 1 << (exp - 1);
	#endif
}

#define BLOCK_SIZE 256

float**			g_scanBlockSums = 0;
int**			g_scanBlockSumsInt = 0;
unsigned int	g_numEltsAllocated = 0;
unsigned int	g_numLevelsAllocated = 0;


void preallocBlockSumsInt (unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {       
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
		{
				level++;
		}
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSumsInt = (int**) malloc(level * sizeof(int*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) cudaCheck ( cudaMalloc((void**) &g_scanBlockSumsInt[level++], numBlocks * sizeof(int)), "Malloc prescanBlockSumsInt g_scanBlockSumsInt");
        numElts = numBlocks;
    } while (numElts > 1);
}


void deallocBlockSumsInt()
{
	if ( g_scanBlockSums != 0x0 ) 
	{
		for (unsigned int i = 0; i < g_numLevelsAllocated; i++) 
		{
			cudaCheck ( cudaFree(g_scanBlockSumsInt[i]), "Malloc deallocBlockSumsInt g_scanBlockSumsInt");
		}
		free( (void**)g_scanBlockSumsInt );
	}

    g_scanBlockSumsInt = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}


void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
	{
        numThreads = blockSize;
	}
    else if (isPowerOfTwo(numElements))
	{
        numThreads = numElements / 2;
	}
    else
	{
        numThreads = floorPow2(numElements);
	}

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock) 
	{
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);            
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // execute the scan
    if (numBlocks > 1) 
	{
        prescanInt <true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSumsInt[level], numThreads * 2, 0, 0);
        if (np2LastBlock) 
		{
            prescanInt <true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be added to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursiveInt (g_scanBlockSumsInt[level], g_scanBlockSumsInt[level], numBlocks, level+1);

        uniformAddInt <<< grid, threads >>> (outArray, g_scanBlockSumsInt[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) 
		{
            uniformAddInt <<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } 
	else if (isPowerOfTwo(numElements)) 
	{
        prescanInt <false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } 
	else 
	{
        prescanInt <false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}



 


