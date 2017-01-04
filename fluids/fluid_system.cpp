#include <assert.h>
#include <stdio.h>
#include <conio.h>

#include "app_perf.h"
#include "fluid_system.h"

#include "fluid_system_host.cuh"
#include "boost\random\uniform_real.hpp"
#include "boost\random.hpp"
#include "boost\lexical_cast.hpp"
#include "boost\random\mersenne_twister.hpp"
#include "boost\random\uniform_on_sphere.hpp"
#include <string>     
#include <sstream>
#include <iostream>
#include <fstream>


//Initialization
FluidSystem::FluidSystem ()
{
	mPos = 0x0;
	mClr = 0x0; 
	mId  = 0x0;
	mClrNorm=0x0;
	mNorm = 0x0;
	mNormLine = 0x0;
	m_isNorm = false;
	m_numFluidParticles =10000;//25000
	m_c = 0;
	SetupDefaultParams ();
}   



void FluidSystem::SetupDefaultParams ()
{

	//////////  dist1 = m_smoothRadius/m_simScale    3.077
	//////////  dist2 = sqrt(pi*m_radius*m_radius/m_numFluidParticles)    1.09
	//////////  dist1/dist2 = 2-3 is good;   

	m_Time = 0;			
	m_DT = 	0.001;				
	m_simScale    =	0.0065f;
	m_restDensity =	120.f;
	m_pmass       =	0.0002f;		
	m_smoothRadius=	0.02f;
	m_gasConstant = 1.5;
	m_aclLimit    =	150.0;		
	m_velLimit    =	5.0;		
	m_gridSize    = m_smoothRadius;
	m_radius      = 55;  
	m_GridMin.Set(-60,-60,-60); 
	m_GridMax.Set(60,60,60);
}




// Allocate particle memory
void FluidSystem::AllocateParticles ( )
{
	mPos      = (Vector3DF*)	malloc ( m_numFluidParticles*sizeof(Vector3DF) );
	mClr      = (DWORD*)		malloc ( m_numFluidParticles*sizeof(DWORD) );  
	mClrNorm  = (DWORD*)		malloc ( m_numFluidParticles*sizeof(DWORD)*2 ); 
	mNorm     = (Vector3DF*)	malloc ( m_numFluidParticles*sizeof(Vector3DF) );
	mNormLine = (Vector3DF*)	malloc ( m_numFluidParticles*sizeof(Vector3DF)*2 );
	mId       = (int*)		    malloc ( m_numFluidParticles*sizeof(int) );
}



//set up grid 
void FluidSystem::SetupGridAllocate ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size )
{
	float m_worldCellsize = cell_size / sim_scale;
	m_GridMin = min;
	m_GridMax = max;
	Vector3DF m_GridSize = m_GridMax-m_GridMin;
	m_GridRes.x = ceil ( m_GridSize.x / m_worldCellsize );		
	m_GridRes.y = ceil ( m_GridSize.y / m_worldCellsize );
	m_GridRes.z = ceil ( m_GridSize.z / m_worldCellsize );

	m_GridSize.x = m_GridRes.x * m_worldCellsize;				
	m_GridSize.y = m_GridRes.y * m_worldCellsize;
	m_GridSize.z = m_GridRes.z * m_worldCellsize;

	m_GridDelta.Set(1.0f/m_worldCellsize,1.0f/m_worldCellsize,1.0f/m_worldCellsize);		
	m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

	m_GridSrch =  3;
}


void FluidSystem::SetupFluid ()
{
	AllocateParticles ();
    SetupFluidSample();
	SetupGridAllocate ( m_GridMin, m_GridMax, m_simScale, m_gridSize);	// Setup grid

	FluidClearCUDA ();
	Sleep ( 300 );
	SetupSysCUDA( m_GridSrch,*(int3*)& m_GridRes,*(float3*)& m_GridDelta,  *(float3*)& m_GridMin, m_GridTotal );
	SetupFluidCUDA ( m_numFluidParticles );
	Sleep ( 300 );
	SetParamCUDA ( m_radius, m_simScale,  m_smoothRadius, m_pmass,m_restDensity, m_gasConstant,m_aclLimit, m_velLimit);
	CopyFluidToCUDA ( (float*) mPos,  (char*) mClr, (int*) mId);	// Initial transfer

}


void FluidSystem::Run (float width, float height)
{
	RunSimulateCUDAFull(m_Time);	 
	m_Time += m_DT;
}



void FluidSystem::RunSimulateCUDAFull (float time)
{

	InsertParticlesCUDA ();
	PrefixSumCellsCUDA ();
	CountingSortFullCUDA ();	

	ComputePressureCUDA();
	ComputeNormalCUDA();
	ComputeForceCUDA ();	
	AdvanceCUDA ( m_Time, m_DT);
	finalize();

	CopyFluidFromCUDA((float*) mPos, (float*) mNorm, (char*) mClr);
	Changecolor(m_c);

	if(m_isNorm==true)
	{
		for(int i = 0;i<m_numFluidParticles*2;i++)
		{
			if(i%2==0)
			{
				mNormLine[i] = mPos[i/2];
			}
			else
			{
				mNormLine[i] = mPos[i/2]+mNorm[i/2];
			}
		}
	}
}



void FluidSystem::SetupFluidSample()
{
    typedef boost::random::mt19937 gen;
    gen rand_gen;
    rand_gen.seed(static_cast<unsigned int>(std::time(0)));
    boost::uniform_on_sphere<float> unif_sphere(3);
	boost::variate_generator<gen&, boost::uniform_on_sphere<float> >    random_on_sphere(rand_gen, unif_sphere);
	uint i = 0;
	do
	{ 
		 std::vector<float> random_sphere_point = random_on_sphere();
		 Vector3DF  p(random_sphere_point[0]*m_radius,random_sphere_point[1]*m_radius,random_sphere_point[2]*m_radius);
		 (mPos+i)->Set(p.x,p.y,p.z);
		 i++;
	
	}while (i<m_numFluidParticles);
}


void FluidSystem::Exit ()
{
 
	free ( mNorm );
	free ( mNormLine );
	free ( mPos );
	free ( mClr ); 
	free ( mClrNorm );
	free ( mId  );
	FluidClearCUDA();
	cudaExit ();
}



void FluidSystem::SetupRender ()
{

	glGenTextures ( 1, (GLuint*) mTex );
	glBindTexture ( GL_TEXTURE_2D, mTex[0] );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);	
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );	
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4);	
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 8, 8, 0, GL_RGB, GL_FLOAT, 0);

	glGenBuffersARB ( 3, (GLuint*) mVBO );
	// Construct a sphere in a VBO
	int udiv = 6;
	int vdiv = 6;
	float du = 180.0 / udiv;
	float dv = 360.0 / vdiv;
	float x,y,z, x1,y1,z1;

	float r = 1.0;

	Vector3DF* buf = (Vector3DF*) malloc ( sizeof(Vector3DF) * (udiv+2)*(vdiv+2)*2 );
	Vector3DF* dat = buf;
	
	int	mSpherePnts = 0;
	for ( float tilt=-90; tilt <= 90.0; tilt += du)
	{
		for ( float ang=0; ang <= 360; ang += dv) 
		{
			x = sin ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			y = cos ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			z = sin ( tilt*DEGtoRAD ) ;
			x1 = sin ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			y1 = cos ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			z1 = sin ( (tilt+du)*DEGtoRAD );
		
			dat->x = x*r;
			dat->y = y*r;
			dat->z = z*r;
			dat++;
			dat->x = x1*r;
			dat->y = y1*r;
			dat->z = z1*r;
			dat++;
			mSpherePnts += 2;
		}
	}
	glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
	glBufferDataARB ( GL_ARRAY_BUFFER_ARB, mSpherePnts*sizeof(Vector3DF), buf, GL_STATIC_DRAW_ARB);
	glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );
	free ( buf );	
	mImg.LoadPng ( "ball32.png" );
	mImg.UpdateTex ();

}




void FluidSystem::DrawParticle ( int p, int r1, int r2, Vector3DF clr )
{
	Vector3DF* ppos = mPos + p;
	DWORD* pclr = mClr + p;
	
	glDisable ( GL_DEPTH_TEST );
	
	glPointSize ( r2 );	
	glBegin ( GL_POINTS );
	glColor3f ( clr.x, clr.y, clr.z ); glVertex3f ( ppos->x, ppos->y, ppos->z );
	glEnd ();

	glEnable ( GL_DEPTH_TEST );
}


void FluidSystem::Draw ( )
{

	glDisable ( GL_LIGHTING );

	glEnable(GL_BLEND); 
				
	glEnable(GL_ALPHA_TEST); 
	glAlphaFunc( GL_GREATER, 0.5 ); 	
	glEnable(GL_POINT_SPRITE_ARB); 		
	float quadratic[] =  { 1.0f, 0.01f, 0.0001f };
	glPointParameterfvARB(  GL_POINT_DISTANCE_ATTENUATION, quadratic );
	glPointParameterfARB( GL_POINT_SIZE_MAX_ARB, 32 );
	glPointParameterfARB( GL_POINT_SIZE_MIN_ARB, 1.0f );

	//// Texture and blending mode
	glEnable ( GL_TEXTURE_2D );
	glBindTexture ( GL_TEXTURE_2D, mImg.getTex() );
	glTexEnvi (GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) ;


	// Point buffers Render - Point Sprites----------fluid
	glPointSize (10);		
	glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
	glBufferDataARB ( GL_ARRAY_BUFFER_ARB, m_numFluidParticles*sizeof(Vector3DF), mPos, GL_DYNAMIC_DRAW_ARB);		
	glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
	glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
	glBufferDataARB ( GL_ARRAY_BUFFER_ARB, m_numFluidParticles*sizeof(uint), mClr, GL_DYNAMIC_DRAW_ARB);
	glColorPointer  ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
	glEnableClientState ( GL_VERTEX_ARRAY );
	glEnableClientState ( GL_COLOR_ARRAY );
	glDrawArrays ( GL_POINTS, 0, m_numFluidParticles );

	
	//// Restore state
	glDisableClientState ( GL_VERTEX_ARRAY );
	glDisableClientState ( GL_COLOR_ARRAY );
	glDisable (GL_POINT_SPRITE_ARB); 
	glDisable ( GL_ALPHA_TEST );
	glDisable ( GL_TEXTURE_2D );
	glDisable (GL_BLEND);
	glDepthMask( GL_TRUE );   
	
		
	if(m_isNorm==true)
	{
		GLuint vertexBufferId;  
		GLuint colorBufferId;
		glGenBuffers(1, &vertexBufferId);   
		glGenBuffers(1, &colorBufferId);

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexBufferId);   
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, m_numFluidParticles*sizeof(Vector3DF)*2, mNormLine, GL_STATIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );	

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, colorBufferId);   
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, m_numFluidParticles*sizeof(uint)*2, mClrNorm, GL_STATIC_DRAW_ARB);	
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 );	

		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );
		glLineWidth(2);
		glDrawArrays ( GL_LINES, 0, m_numFluidParticles*2 );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );

	}
}









