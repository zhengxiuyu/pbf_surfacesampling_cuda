
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "app_opengl.h"
#include "app_perf.h"

#include "fluid_system_host.cuh"	
#include "fluid_system.h"

float frameTime;
int frameFPS;
Time timer; 

FluidSystem		psys;
Camera3D		cam;

bool    bPause = false;

// View matricies
float model_matrix[16];					// Model matrix (M)

// Mouse control
#define DRAG_OFF		0				// mouse states
#define DRAG_LEFT		1
#define DRAG_RIGHT		2
int		last_x = -1, last_y = -1;		// mouse vars
int		mode = 0;
int		dragging = 0;


// Different things we can move around
#define MODE_CAM		0
#define MODE_CAM_TO		1



void drawScene ( float* viewmat )
{
	psys.Draw ();				// Draw particles		
}



void display () 
{
	PERF_PUSH ( " " );	

	// Do simulation!
	if ( !bPause ) psys.Run (window_width, window_height);


	// Clear frame buffer
	glClearColor( 1.0f, 1.0f, 1.0f, 1.0 );
	
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glDisable ( GL_LIGHTING );
	//psys.DrawNorm();


	// Compute camera view
	cam.updateMatricies ();
	glMatrixMode ( GL_PROJECTION );
	glLoadMatrixf ( cam.getProjMatrix().GetDataF() );
	
	// Draw 3D	
	glMatrixMode ( GL_MODELVIEW );
	glLoadMatrixf ( cam.getViewMatrix().GetDataF() );

	psys.Draw ();	

	drawGui ();
	draw2D ();

	// Swap buffers
	SwapBuffers ( g_hDC );

	frameTime = PERF_POP ();	
	frameFPS = int(1000.0 / frameTime);
	app_printf("frameFPS:%d\n",frameFPS); //加上这一句程序不执行
}

void reshape ( int width, int height ) 
{
  // set window height and width
  glViewport( 0, 0, width, height );    
  setview2D ( width, height );
}



void keyboard_func ( unsigned char key, int x, int y )
{
	Vector3DF fp = cam.getPos ();
	Vector3DF tp = cam.getToPos ();

	switch( key ) 
	{
	case '0': psys.setColor(0); break;
	case '1': psys.setColor(1); break;
	case '2': psys.setColor(2); break;
	case 't': case 'T':		psys.SetupFluid (); break;  
	case 'o': case 'O':		psys.m_isNorm = !psys.m_isNorm ; break;
	case ' ': bPause = !bPause;	break;		// pause
	case 27:  exit( 0 ); break;

	//Camera
	case 'c':case 'C':	mode = (mode+1)%2;	break;
	case 'a': case 'A':		cam.setToPos( tp.x - 1, tp.y, tp.z ); break;
	case 'd': case 'D':		cam.setToPos( tp.x + 1, tp.y, tp.z ); break;
	case 'w': case 'W':		cam.setToPos( tp.x, tp.y - 1, tp.z ); break;
	case 's': case 'S':		cam.setToPos( tp.x, tp.y + 1, tp.z ); break;
	case 'q': case 'Q':		cam.setToPos( tp.x, tp.y, tp.z + 1 ); break;
	case 'z': case 'Z':		cam.setToPos( tp.x, tp.y, tp.z - 1 ); break;
	default:
	break;
  }
}



void mouse_click_func ( int button, int state, int x, int y )
{
	if( state == GLUT_DOWN ) 
	{
		// Handle 2D gui interaction first
		if ( guiMouseDown ( x, y ) ) return;

		if ( button == GLUT_LEFT_BUTTON )		dragging = DRAG_LEFT;
		else if ( button == GLUT_RIGHT_BUTTON ) dragging = DRAG_RIGHT;	
		last_x = x;
		last_y = y;	

	} 
	else if ( state==GLUT_UP ) 
	{
		dragging = DRAG_OFF;
	}
}

void mouse_move_func ( int x, int y )
{

}

void mouse_drag_func ( int x, int y )
{
	int dx = x - last_x;
	int dy = y - last_y;

	// Handle GUI interaction in nvGui by calling guiMouseDrag
	if ( guiMouseDrag ( x, y ) ) return;

	switch ( mode ) 
	{

		case MODE_CAM:
			if ( dragging == DRAG_LEFT ) 
			{
				cam.moveOrbit ( -dx, dy, 0, 0);
			} else if ( dragging == DRAG_RIGHT ) 
			{
				cam.moveOrbit ( 0, 0, 0, dy*0.15f );
			} 
			break;
		case MODE_CAM_TO:
			if ( dragging == DRAG_LEFT ) 
			{
				cam.moveToPos ( dx*0.1f, 0, dy*0.1f );
			} else if ( dragging == DRAG_RIGHT )
			{
				cam.moveToPos ( 0, dy*0.1f, 0 );
			}
			break;	
	}

	last_x = x;
	last_y = y;
}


void idle_func ()
{

}

void init ()
{
	cam.setOrbit ( Vector3DF(0,0,0), Vector3DF(0,0,0.5), 124, 100 );
	cam.setFov ( 295 );
	cam.updateMatricies ();
	psys.SetupFluid ();
}


void initialize ()
{

	cudaInit ();


	PERF_INIT ( true );
	PERF_SET ( true, 0, true, "" );
			
	addGui (  20,   20, 200, 12, "Frame Time - FPS ",	GUI_PRINT,  GUI_INT,	&frameFPS, 0, 0 );					
	addGui (  20,   35, 200, 12, "Frame Time - msec ",	GUI_PRINT,  GUI_FLOAT,	&frameTime, 0, 0 );							
	init2D ( "arial_12" );		// specify font file (.bin/tga)
	setText ( 1.0, -0.5 );		// scale by 0.5, kerning adjust -0.5 pixels
	setview2D ( window_width, window_height );
	setorder2D ( true, -0.001 );
	
	init();		
	psys.SetupRender ();
}


void shutdown ()
{
	
}
