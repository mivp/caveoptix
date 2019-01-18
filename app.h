#ifndef __OPTIX_APP_H
#define __OPTIX_APP_H

#ifdef OMEGALIB_MODULE
#include <omegaGl.h>
#else
#include "stdapp/GLInclude.h"
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

struct nvVertex {
	nvVertex(float x1, float y1, float z1, float tx1, float ty1, float tz1) { x=x1; y=y1; z=z1; tx=tx1; ty=ty1; tz=tz1; }
	float	x, y, z;
	float	nx, ny, nz;
	float	tx, ty, tz;
};
struct nvFace {
	nvFace(unsigned int x1, unsigned int y1, unsigned int z1) { a=x1; b=y1; c=z1; }
	unsigned int  a, b, c;
};

typedef optix::float3 float3;
typedef optix::float4 float4;

class OptixApp {

protected:
    void checkGL( char* msg );
    void initGL();
    void initScreenQuadGL();
    void createScreenQuadGL ( int* glid, int w, int h );
    void renderScreenQuadGL ( int glid1, int glid2, float x1, float y1, float x2, float y2, char inv1, char inv2 );
    void renderScreenQuadGL( int glid, char inv1, int w, int h );

    // optix
    optix::float4 make_plane( optix::float3 n, optix::float3 p );
    void createGeometry();
    void setupLights();
    void setupCamera();


public:
    OptixApp();
    ~OptixApp();

    void init(int w, int h);
    void display(const float V[16], const float P[16], const float pos[3]);

    
private:
    bool    m_initialized;
    int     m_framecount;

    int     m_width, m_height;
    int		m_screenquad_prog;
    int		m_screenquad_vshader;
    int		m_screenquad_fshader;
    int		m_screenquad_vbo[3];
    int		m_screenquad_utex1;
    int		m_screenquad_utex2;
    int		m_screenquad_utexflags;
    int		m_screenquad_ucoords;
    int		m_screenquad_uscreen;

    unsigned int	gl_screen_tex; 

    //Optix
    int             tutorial_number;
    optix::Context  context;
    bool            use_pbo;
    std::string     texture_path;
    std::string     tutorial_ptx;
    // Camera state
    optix::float3       camera_up;
    optix::float3       camera_lookat;
    optix::float3       camera_eye;
    optix::Matrix4x4    camera_rotate;
    unsigned int        gl_tex_id;
};


#endif