#ifndef DEF_HOST_CUDA
	#define DEF_HOST_CUDA

	#include <vector_types.h>
	#include <driver_types.h>			// for cudaStream_t
#include<vector>
	typedef unsigned int		uint;
	typedef unsigned short		ushort;
	typedef unsigned char		uchar;

	extern "C"
	{
	void cudaInit();
	void cudaExit();

	void FluidClearCUDA ();

	void SetupSysCUDA( int gsrch, int3 res,  float3 delta, float3 gmin,  int total);
	void SetupFluidCUDA (int num);
	void SetParamCUDA ( float r, float ss, float sr, float mass, float rest,  float istiff,   float al, float vl);
	void CopyFluidToCUDA ( float* pos,   char* clr, int* id);
	
	void finalize ();
	void Changecolor(int c);
	void CopyFluidFromCUDA(float* pos,float* norm, char* clr);
	void InsertParticlesCUDA ();	
	void PrefixSumCellsCUDA ();
	void CountingSortFullCUDA ();
	void ComputePressureCUDA ();
	void ComputeForceCUDA ();	
	void ComputeNormalCUDA();
	void AdvanceCUDA ( float time, float dt);

	void preallocBlockSumsInt(unsigned int num);
	void deallocBlockSumsInt();
	void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level);

	}

#endif