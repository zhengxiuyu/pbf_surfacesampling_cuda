
#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>
	#include <vector>

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS			16
	#define LOG_NUM_BANKS		4
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295



	// Fluid Parameters (stored on both host and device)
	
	struct sysParams
	{
		float		    poly6kern, poly6kernGra,poly6kernLap, spikykern, lapkern, cohesion;
		float		    pintstiff;
		float3			gridMin, gridDelta;
		float			psmoothradius, psimscale;
		float           radius;
		float			d2,r2, rd2;
		int3			gridRes, gridScanMax;
		int				gridAdj[64];
		int				gridSrch, gridTotal, gridAdjCnt;
		int				gridThreads, gridBlocks;	
		int				szGrid;
	};
	
	struct fluidParams 
	{
		int             pnum;
		int				numThreads, numBlocks;
		float			pmass, prest_dens;
		float			AL, AL2, VL, VL2;
	};

	

	
	// Temporary sort buffer offsets
	#define BUF_POS			0									//0
	#define BUF_VEL			(sizeof(float3))					//12
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))			//24
	#define BUF_CLR			(BUF_VELEVAL + sizeof(float3))			//64
	#define BUF_ID		    (BUF_CLR + sizeof(uint))			//64
	#define BUF_ISBOUNDARY  (BUF_ID + sizeof(int))			//64
	#define BUF_NORM        (BUF_ISBOUNDARY + sizeof(bool))	
	#define BUF_GCELL		(BUF_NORM + sizeof(float3))			//56
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))			//60


	struct fluidBufList 
	{
		float3*			mpos;
		float3*			mvel;
		float*          mMass;
		float3*			mveleval;
		uint*			mclr;	
		int*            mId;
		bool*           mIsBoundary;
		float3*         mNorm;
		uint*			mgcell;
		uint*			mgndx;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;
		float*          mperDensity;
		float3*         mPoPos;

	
		char*			msortbuf;
		uint*			mgrid;	
		int*			mgridcnt;
		int*			mgridoff;
	};




	#ifndef CUDA_KERNEL		

		// Declare kernel functions that are available to the host.
		// These are defined in kern.cu, but declared here so host.cu can call them.

		__global__ void insertParticles ( fluidBufList buf, int pnum );
		__global__ void countingSortFull ( fluidBufList buf, int pnum );	
		__global__ void computePressure ( fluidBufList buf, int pnum, float restDen );	
		__global__ void computeDensity(fluidBufList buf, int pnum);
		__global__ void computeForce ( fluidBufList buf, int pnum,float dens);
		__global__ void computeNormal( fluidBufList buf, int pnum);
		__global__ void advanceParticles ( float time, float dt, fluidBufList buf, int numPnts);
		__global__ void updateParticles ( fluidBufList buf, int numPnts );
		__global__ void updateColor(fluidBufList buf, int numPnts , int c);
		void updateSimParams ( fluidParams* cpufp);
		void updateSysParams(sysParams* sysp);

		// Prefix Sum
		#include "prefix_sum.cu" //为什么include .cu文件？ .cu没有头文件，其中包含的都是__device__的函数？？？
		// NOTE: Template functions must be defined in the header

		template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) 
		{
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ int s_dataInt [];
			loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
			prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums); 
			storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
		}
		__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);	
		__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);	
	#endif
	

	
#endif
