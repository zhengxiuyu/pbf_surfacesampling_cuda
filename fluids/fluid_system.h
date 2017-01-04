#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS

	#include <iostream>
	#include <string>    
	#include <sstream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	typedef unsigned int		uint;	
	#include "xml_settings.h"

	class FluidSystem 
	{
	public:
		FluidSystem ();
		
		// Rendering
		void Draw ();
		void DrawGrid ();
		void DrawParticle ( int p, int r1, int r2, Vector3DF clr2 );
	
		
		// Setup
		void SetupFluid ();
		void SetupRender ();
		void AllocateParticles ();
		void SetupDefaultParams ();
		void SetupFluidSample();
		void SetupGridAllocate ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size);

		// Simulation
		void Run (float w, float h);
		void RunSimulateCUDAFull (float time);
		void Exit ();
	
		// Parameters			
		int getSmapleNum()   {return m_numFluidParticles;}	
		void setColor(int c) {m_c = c;}
		bool m_isNorm;

	private:
		float					m_DT;
		float					m_Time;	
		int						m_c;					//color
		float                   m_radius;				//sampling area

		// Particle Buffers
		Vector3DF*				mPos;
		DWORD*					mClr;
		DWORD*					mClrNorm;
		Vector3DF*				mNorm;
		Vector3DF*				mNormLine;
        int*                    mId;

		// Fluid Acceleration Grid
		uint*					m_Grid;
		float					m_gridSize;
		int						m_GridTotal;			// total # cells
		Vector3DI				m_GridRes;				// resolution in each axis
		Vector3DF				m_GridMin;				// volume of grid (may not match domain volume exactly)
		Vector3DF				m_GridMax;		
		Vector3DF				m_GridDelta;
		int						m_GridSrch;

		int						m_numFluidParticles;	
		float					m_restDensity; 
		float					m_simScale;
		float					m_smoothRadius;
		float					m_pmass;
		float					m_gasConstant;
		float					m_velLimit;
		float					m_aclLimit;

		//draw
		int						mVBO[3];
		int						mTex[1];
		nvImg					mImg;
	};	

#endif
