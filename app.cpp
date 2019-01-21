#include "app.h"

#include <string>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <sstream>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>
#include <memory.h>

#include "sutil.h"
#include "commonStructs.h"
#include "random.h"

using namespace std;

const char* const SAMPLE_NAME = "optixTutorial";

static float rand_range(float min, float max)
{
    static unsigned int seed = 0u;
    return min + (max - min) * rnd(seed);
}

static const char *g_screenquad_vert =
	"#version 440 core\n"
	"layout(location = 0) in vec3 vertex;\n"
	"layout(location = 1) in vec3 normal;\n"
	"layout(location = 2) in vec3 texcoord;\n"
	"uniform vec4 uCoords;\n"
	"uniform vec2 uScreen;\n"
	"out vec3 vtc;\n"
	"void main() {\n"
	"   vtc = texcoord*0.5+0.5;\n"
	"   gl_Position = vec4( -1.0 + (uCoords.x/uScreen.x) + (vertex.x+1.0f)*(uCoords.z-uCoords.x)/uScreen.x,\n"
	"                       -1.0 + (uCoords.y/uScreen.y) + (vertex.y+1.0f)*(uCoords.w-uCoords.y)/uScreen.y,\n"
	"                       0.0f, 1.0f );\n"
	"}\n";

static const char *g_screenquad_frag =
	"#version 440\n"
	"uniform sampler2D uTex1;\n"
	"uniform sampler2D uTex2;\n"
	"uniform int uTexFlags;\n"
	"in vec3 vtc;\n"
	"out vec4 outColor;\n"
	"void main() {\n"
	"   vec4 op1 = ((uTexFlags & 0x01)==0) ? texture ( uTex1, vtc.xy) : texture ( uTex1, vec2(vtc.x, 1.0-vtc.y));\n"
	"   if ( (uTexFlags & 0x02) != 0 ) {\n"
	"		vec4 op2 = ((uTexFlags & 0x04)==0) ? texture ( uTex2, vtc.xy) : texture ( uTex2, vec2(vtc.x, 1.0-vtc.y));\n"
	"		outColor = vec4( op1.xyz*(1.0-op2.w) + op2.xyz * op2.w, 1 );\n"
	"   } else { \n"
	"		outColor = vec4( op1.xyz, 1 );\n"
	"   }\n"
	"}\n";


OptixApp::OptixApp(): m_initialized(false), m_framecount(0),
						context(0), use_pbo(true), gl_tex_id(0),
                        gl_screen_tex(0)
{
    tutorial_number = 1;
    tutorial_ptx = sutil::getPtxString(NULL, "tutorial1.cu");
}

OptixApp::~OptixApp() {
	if(context) {
		context->destroy();
		context = 0;
	}
}

void OptixApp::init(int w, int h) {

    m_width = w;
    m_height = h;

    // GL
    initGL();

	// set up context
	context = optix::Context::create();
	context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 4640 );

	// Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 100 );
    context["radiance_ray_type"]->setUint( 0 );
    context["shadow_ray_type"]->setUint( 1 );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["importance_cutoff"]->setFloat( 0.01f );
    context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

    // output buffer
    optix::Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Ray generation program
#ifdef OMEGALIB_MODULE
    const std::string camera_name = "pinhole_camera_omegalib";
#else
    const std::string camera_name = "pinhole_camera";
#endif
    optix::Program ray_gen_program = context->createProgramFromPTXString( tutorial_ptx, camera_name );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    optix::Program exception_program = context->createProgramFromPTXString( tutorial_ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    const std::string miss_name = tutorial_number >= 5 ? "envmap_miss" : "miss";
    context->setMissProgram( 0, context->createProgramFromPTXString( tutorial_ptx, miss_name ) );
    const float3 default_color = optix::make_float3(1.0f, 1.0f, 1.0f);
    const std::string texpath = "./data/" + std::string( "CedarCity.hdr" );
    context["envmap"]->setTextureSampler( sutil::loadTexture( context, texpath, default_color) );
    context["bg_color"]->setFloat( optix::make_float3( 0.34f, 0.55f, 0.85f ) );

    // 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].
    const int tex_width  = 64;
    const int tex_height = 64;
    const int tex_depth  = 64;
    optix::Buffer noiseBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
    float *tex_data = (float *) noiseBuffer->map();

    // Random noise in range [0, 1]
    for (int i = tex_width * tex_height * tex_depth;  i > 0; i--) {
        // One channel 3D noise in [0.0, 1.0] range.
        *tex_data++ = rand_range(0.0f, 1.0f);
    }
    noiseBuffer->unmap(); 

    // Noise texture sampler
    optix::TextureSampler noiseSampler = context->createTextureSampler();

    noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
    noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
    noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    noiseSampler->setMaxAnisotropy(1.0f);
    noiseSampler->setMipLevelCount(1);
    noiseSampler->setArraySize(1);
    noiseSampler->setBuffer(0, 0, noiseBuffer);

    context["noise_texture"]->setTextureSampler(noiseSampler);

    // others
    createGeometry();
    setupLights();
    setupCamera();

    //createScreenQuadGL ( &gl_screen_tex, m_width, m_height );
	
	m_initialized = true;
}

// Converts the buffer format to gl format
GLenum glFormatFromBufferFormat(bufferPixelFormat pixel_format, RTformat buffer_format)
{
    if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
    {
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_BGRA;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGRA;
        default:
            cout << "Unknown buffer pixel format" << endl;
            exit(1);
            //throw Exception("Unknown buffer pixel format");
        }
    }
    else if (buffer_format == RT_FORMAT_FLOAT4)
    {
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGRA;
        default:
            cout << "Unknown buffer pixel format" << endl;
            exit(1);
            //throw Exception("Unknown buffer pixel format");
        }
    }
    else if (buffer_format == RT_FORMAT_FLOAT3)
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_RGB;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGB;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGR;
        default:
            cout << "Unknown buffer pixel format" << endl;
            exit(1);
            //throw Exception("Unknown buffer pixel format");
        }
    else if (buffer_format == RT_FORMAT_FLOAT)
        return GL_LUMINANCE;
    else {
        cout << "Unknown buffer pixel format" << endl;
        exit(1);
        //throw Exception("Unknown buffer format");
    }
}

#define DEG2RAD(x) (x * 0.01745329251994329575)
#define RAD2DEG(x) (x * 57.2957795131)

void OptixApp::display(const float V[16], const float P[16], const float campos[3]) {
	if(!m_initialized) return;

    // update camera
    const float vfov = 45.0f;
    const float aspect_ratio = static_cast<float>(m_width) /
                               static_cast<float>(m_height);

    float3 camera_u, camera_v, camera_w;
    int TYPE = 0; // 0: calculate (U, V, W) from (MV, P, campos)

    if(TYPE == 0) {
        float b = P[5];
        float FOV = 2.0f * (float)atan(1.0f/b);
        float focal = 1 / tan(FOV/2);
        
        camera_eye = optix::make_float3(campos[0], campos[1], campos[2]);
        //camera_eye = optix::make_float3(V[12], V[13], V[14]);
        camera_u = optix::make_float3(V[0], V[4], V[8]);
        camera_v = -1*optix::make_float3(V[1], V[5], V[9]);
        camera_w = -1*optix::make_float3(V[2], V[6], V[10]);
        
        float fovY = 0.5 * FOV;
        float fovX = atan(tan(FOV)*aspect_ratio);
        float ulen = focal * tan(FOV); // * aspect_ratio;
        float vlen = focal * tan(fovY);
        camera_u = ulen * camera_u;
        camera_v = vlen * camera_v;
        camera_w = focal * camera_w;
        
        if(m_framecount < 1) {
            cout << "FOV: " << RAD2DEG(FOV) << " fovY: " << RAD2DEG(fovY) << " fovX: " << RAD2DEG(fovX) << " focal: " << focal << endl;
        }
    }
    else {
        sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

        if(m_framecount < 1) {
            cout << "u: " << camera_u << endl;
            cout << "v: " << camera_v << endl;
            cout << "w: " << camera_w << endl;
        }
        
        const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
                normalize( camera_u ),
                normalize( camera_v ),
                normalize( -camera_w ),
                camera_lookat);
        const optix::Matrix4x4 frame_inv = frame.inverse();
        // Apply camera rotation twice to match old SDK behavior
        const optix::Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

        camera_eye    = optix::make_float3( trans*optix::make_float4( camera_eye,    1.0f ) );
        camera_lookat = optix::make_float3( trans*optix::make_float4( camera_lookat, 1.0f ) );
        camera_up     = optix::make_float3( trans*optix::make_float4( camera_up,     0.0f ) );

        sutil::calculateCameraVariables(
                camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
                camera_u, camera_v, camera_w, true );

        camera_rotate = optix::Matrix4x4::identity();
    }

    if(m_framecount < 1) {
        cout << "eye: " << camera_eye << endl;
        cout << "u: " << camera_u << endl;
        cout << "v: " << camera_v << endl;
        cout << "w: " << camera_w << endl;
    }
    
    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

    // render
    context->launch( 0, m_width, m_height );
    optix::Buffer buffer = context[ "output_buffer" ]->getBuffer();
    
    // Query buffer information
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );
    uint32_t width  = static_cast<int>(buffer_width_rts);
    uint32_t height = static_cast<int>(buffer_height_rts);
    RTformat buffer_format = buffer->getFormat();
    //cout << width << " " << height << endl;

    if( !gl_screen_tex )
    {
        glGenTextures( 1, &gl_screen_tex );
        glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

        // Change these to GL_LINEAR for super- or sub-sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        cout << "gl_screen_tex: " << gl_screen_tex << endl;
    }

    glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

    // send PBO or host-mapped image data to texture
    const unsigned pboId = buffer->getGLBOId();
    GLvoid* imageData = 0;
    if( pboId )
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboId );
    else
        imageData = buffer->map( 0, RT_BUFFER_MAP_READ );

    RTsize elmt_size = buffer->getElementSize();
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLenum pixel_format = glFormatFromBufferFormat(BUFFER_PIXEL_FORMAT_DEFAULT, buffer_format);

    if( buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, pixel_format, GL_UNSIGNED_BYTE, imageData);
    else if(buffer_format == RT_FORMAT_FLOAT4)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else if(buffer_format == RT_FORMAT_FLOAT3)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else if(buffer_format == RT_FORMAT_FLOAT)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else {
        cout << "Unkown buffer format" << endl;
        exit(1);
        // throw Exception( "Unknown buffer format" );
    }

    if( pboId )
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    else
        buffer->unmap();

    renderScreenQuadGL ( gl_screen_tex, 0, width, height );
    
	m_framecount++;
}

// for Omegalib CAVE
void OptixApp::display(const float cam_pos[3], const float cam_ori[4], const float head_off[3], 
                const float tl[3], const float bl[3], const float br[3]) {

    if(!m_initialized) return;

    optix::float3 camera_position = optix::make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
    optix::float4 camera_orientation = optix::make_float4(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3]);
    optix::float3 head_offset = optix::make_float3(head_off[0], head_off[1], head_off[2]);
    optix::float3 tile_tl = optix::make_float3(tl[0], tl[1], tl[2]);
    optix::float3 tile_bl = optix::make_float3(bl[0], bl[1], bl[2]);
    optix::float3 tile_br = optix::make_float3(br[0], br[1], br[2]);

    // DEBUG
    if(m_framecount == 0) {
        cout << "cam pos: " << camera_position << endl;
        cout << "cam ori: " << camera_orientation << endl;
        cout << "head offset: " << head_offset << endl;
        cout << "tile tl: " << tile_tl << " bl: " << tile_bl << " br: " << tile_br << endl;
    }

    context["camera_position"]->setFloat( camera_position );
    context["camera_orientation"  ]->setFloat( camera_orientation );
    context["head_offset"  ]->setFloat( head_offset );
    context["tile_tl"  ]->setFloat( tile_tl );
    context["tile_bl"  ]->setFloat( tile_bl );
    context["tile_br"  ]->setFloat( tile_br );

    // render
    context->launch( 0, m_width, m_height );
    optix::Buffer buffer = context[ "output_buffer" ]->getBuffer();
    
    // Query buffer information
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );
    uint32_t width  = static_cast<int>(buffer_width_rts);
    uint32_t height = static_cast<int>(buffer_height_rts);
    RTformat buffer_format = buffer->getFormat();
    //cout << width << " " << height << endl;

    if( !gl_screen_tex )
    {
        glGenTextures( 1, &gl_screen_tex );
        glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

        // Change these to GL_LINEAR for super- or sub-sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        cout << "gl_screen_tex: " << gl_screen_tex << endl;
    }

    glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

    // send PBO or host-mapped image data to texture
    const unsigned pboId = buffer->getGLBOId();
    GLvoid* imageData = 0;
    if( pboId )
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboId );
    else
        imageData = buffer->map( 0, RT_BUFFER_MAP_READ );

    RTsize elmt_size = buffer->getElementSize();
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLenum pixel_format = glFormatFromBufferFormat(BUFFER_PIXEL_FORMAT_DEFAULT, buffer_format);

    if( buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, pixel_format, GL_UNSIGNED_BYTE, imageData);
    else if(buffer_format == RT_FORMAT_FLOAT4)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else if(buffer_format == RT_FORMAT_FLOAT3)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else if(buffer_format == RT_FORMAT_FLOAT)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else {
        cout << "Unkown buffer format" << endl;
        exit(1);
        // throw Exception( "Unknown buffer format" );
    }

    if( pboId )
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    else
        buffer->unmap();

    renderScreenQuadGL ( gl_screen_tex, 0, width, height );

    m_framecount++;
}

optix::float4 OptixApp::make_plane( optix::float3 n, optix::float3 p ) {
    n = normalize(n);
    float d = -dot(n, p);
    return optix::make_float4( n, d );
}

void OptixApp::createGeometry() {
    const char *ptx = sutil::getPtxString( NULL, "box.cu" );
    optix::Program box_bounds    = context->createProgramFromPTXString( ptx, "box_bounds" );
    optix::Program box_intersect = context->createProgramFromPTXString( ptx, "box_intersect" );

    // Create box
    optix::Geometry box = context->createGeometry();
    box->setPrimitiveCount( 1u );
    box->setBoundingBoxProgram( box_bounds );
    box->setIntersectionProgram( box_intersect );
    box["boxmin"]->setFloat( -2.0f, 0.0f, -2.0f );
    box["boxmax"]->setFloat(  2.0f, 7.0f,  2.0f );

    // Create chull
    optix::Geometry chull = 0;
    if( tutorial_number >= 9){
        chull = context->createGeometry();
        chull->setPrimitiveCount( 1u );
        chull->setBoundingBoxProgram( context->createProgramFromPTXString( tutorial_ptx, "chull_bounds" ) );
        chull->setIntersectionProgram( context->createProgramFromPTXString( tutorial_ptx, "chull_intersect" ) );
        optix::Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
        plane_buffer->setFormat(RT_FORMAT_FLOAT4);
        int nsides = 6;
        plane_buffer->setSize( nsides + 2 );
        float4* chplane = (float4*)plane_buffer->map();
        float radius = 1;
        float3 xlate = optix::make_float3(-1.4f, 0, -3.7f);

        for(int i = 0; i < nsides; i++){
            float angle = float(i)/float(nsides) * M_PIf * 2.0f;
            float x = cos(angle);
            float y = sin(angle);
            chplane[i] = make_plane( optix::make_float3(x, 0, y), optix::make_float3(x*radius, 0, y*radius) + xlate);
        }
        float min = 0.02f;
        float max = 3.5f;
        chplane[nsides + 0] = make_plane( optix::make_float3(0, -1, 0), optix::make_float3(0, min, 0) + xlate);
        float angle = 5.f/nsides * M_PIf * 2;
        chplane[nsides + 1] = make_plane( optix::make_float3(cos(angle),  .7f, sin(angle)), optix::make_float3(0, max, 0) + xlate);
        plane_buffer->unmap();
        chull["planes"]->setBuffer(plane_buffer);
        chull["chull_bbmin"]->setFloat(-radius + xlate.x, min + xlate.y, -radius + xlate.z);
        chull["chull_bbmax"]->setFloat( radius + xlate.x, max + xlate.y,  radius + xlate.z);
    }

    // Floor geometry
    optix::Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    ptx = sutil::getPtxString( NULL, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    float3 anchor = optix::make_float3( -64.0f, 0.01f, -64.0f );
    float3 v1 = optix::make_float3( 128.0f, 0.0f, 0.0f );
    float3 v2 = optix::make_float3( 0.0f, 0.0f, 128.0f );
    float3 normal = cross( v2, v1 );
    normal = normalize( normal );
    float d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    optix::float4 plane = optix::make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    // Materials
    std::string box_chname;
    if(tutorial_number >= 8){
        box_chname = "box_closest_hit_radiance";
    } else if(tutorial_number >= 3){
        box_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        box_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        box_chname = "closest_hit_radiance1";
    } else {
        box_chname = "closest_hit_radiance0";
    }

    optix::Material box_matl = context->createMaterial();
    optix::Program box_ch = context->createProgramFromPTXString( tutorial_ptx, box_chname.c_str() );
    box_matl->setClosestHitProgram( 0, box_ch );
    if( tutorial_number >= 3) {
        optix::Program box_ah = context->createProgramFromPTXString( tutorial_ptx, "any_hit_shadow" );
        box_matl->setAnyHitProgram( 1, box_ah );
    }
    box_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
    box_matl["Kd"]->setFloat( 0.6f, 0.7f, 0.8f );
    box_matl["Ks"]->setFloat( 0.8f, 0.9f, 0.8f );
    box_matl["phong_exp"]->setFloat( 88 );
    box_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );

    std::string floor_chname;
    if(tutorial_number >= 7){
        floor_chname = "floor_closest_hit_radiance";
    } else if(tutorial_number >= 6){
        floor_chname = "floor_closest_hit_radiance5";
    } else if(tutorial_number >= 4){
        floor_chname = "floor_closest_hit_radiance4";
    } else if(tutorial_number >= 3){
        floor_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        floor_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        floor_chname = "closest_hit_radiance1";
    } else {
        floor_chname = "closest_hit_radiance0";
    }

    optix::Material floor_matl = context->createMaterial();
    optix::Program floor_ch = context->createProgramFromPTXString( tutorial_ptx, floor_chname.c_str() );
    floor_matl->setClosestHitProgram( 0, floor_ch );
    if(tutorial_number >= 3) {
        optix::Program floor_ah = context->createProgramFromPTXString( tutorial_ptx, "any_hit_shadow" );
        floor_matl->setAnyHitProgram( 1, floor_ah );
    }
    floor_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.1f );
    floor_matl["Kd"]->setFloat( 194/255.f*.6f, 186/255.f*.6f, 151/255.f*.6f );
    floor_matl["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
    floor_matl["reflectivity"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["reflectivity_n"]->setFloat( 0.05f, 0.05f, 0.05f );
    floor_matl["phong_exp"]->setFloat( 88 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.02f );

    // Glass material
    optix::Material glass_matl;
    if( chull.get() ) {
        optix::Program glass_ch = context->createProgramFromPTXString( tutorial_ptx, "glass_closest_hit_radiance" );
        const std::string glass_ahname = tutorial_number >= 10 ? "glass_any_hit_shadow" : "any_hit_shadow";
        optix::Program glass_ah = context->createProgramFromPTXString( tutorial_ptx, glass_ahname.c_str() );
        glass_matl = context->createMaterial();
        glass_matl->setClosestHitProgram( 0, glass_ch );
        glass_matl->setAnyHitProgram( 1, glass_ah );

        glass_matl["importance_cutoff"]->setFloat( 1e-2f );
        glass_matl["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
        glass_matl["fresnel_exponent"]->setFloat( 3.0f );
        glass_matl["fresnel_minimum"]->setFloat( 0.1f );
        glass_matl["fresnel_maximum"]->setFloat( 1.0f );
        glass_matl["refraction_index"]->setFloat( 1.4f );
        glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
        glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
        glass_matl["refraction_maxdepth"]->setInt( 100 );
        glass_matl["reflection_maxdepth"]->setInt( 100 );
        float3 extinction = optix::make_float3(.80f, .89f, .75f);
        glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
        glass_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
    }

    // Create GIs for each piece of geometry
    std::vector<optix::GeometryInstance> gis;
    gis.push_back( context->createGeometryInstance( box, &box_matl, &box_matl+1 ) );
    gis.push_back( context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
    if(chull.get())
        gis.push_back( context->createGeometryInstance( chull, &glass_matl, &glass_matl+1 ) );
    
    // Place all in group
    optix::GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
    geometrygroup->setChild( 0, gis[0] );
    geometrygroup->setChild( 1, gis[1] );
    if(chull.get()) {
        geometrygroup->setChild( 2, gis[2] );
    }
    geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

    context["top_object"]->set( geometrygroup );
    context["top_shadower"]->set( geometrygroup );
}

void OptixApp::setupLights() {
    BasicLight lights[] = { 
        { optix::make_float3( -5.0f, 60.0f, -16.0f ), optix::make_float3( 1.0f, 1.0f, 1.0f ), 1 }
    };

    optix::Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}

void OptixApp::setupCamera() {
    camera_eye    = optix::make_float3( 7.0f, 9.2f, -6.0f );
    camera_lookat = optix::make_float3( 0.0f, 4.0f,  0.0f );
    camera_up     = optix::make_float3( 0.0f, 1.0f,  0.0f );

    camera_rotate  = optix::Matrix4x4::identity();
}


void OptixApp::checkGL( char* msg ) {
	GLenum errCode;
    //const GLubyte* errString;
    errCode = glGetError();
    if (errCode != GL_NO_ERROR) {
		const char * message = "";
		switch( errCode )
		{
		case GL_INVALID_ENUM:
			message = "Invalid enum";
			break;
		case GL_INVALID_VALUE:
			message = "Invalid value";
			break;
		case GL_INVALID_OPERATION:
			message = "Invalid operation";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			message = "Invalid framebuffer operation";
			break;
		case GL_OUT_OF_MEMORY:
			message = "Out of memory";
			break;
		default:
			message = "Unknown error";
		}

        //printf ( "%s, ERROR: %s\n", msg, gluErrorString(errCode) );
		printf ( "%s %s\n", msg, message );
    }
}


void OptixApp::initGL() {
	initScreenQuadGL();
	glFinish();
}

void OptixApp::initScreenQuadGL() {
	int status;
	int maxLog = 65536, lenLog;
	char log[65536];

	// Create a screen-space shader
	m_screenquad_prog = (int)glCreateProgram();
	GLuint vShader = (int)glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vShader, 1, (const GLchar**)&g_screenquad_vert, NULL);
	glCompileShader(vShader);
	glGetShaderiv(vShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		glGetShaderInfoLog(vShader, maxLog, &lenLog, log);
		printf("*** Compile Error in init_screenquad vShader\n");
		printf("  %s\n", log);
	}

	GLuint fShader = (int)glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fShader, 1, (const GLchar**)&g_screenquad_frag, NULL);
	glCompileShader(fShader);
	glGetShaderiv(fShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		glGetShaderInfoLog(fShader, maxLog, &lenLog, log);
		printf("*** Compile Error in init_screenquad fShader\n");
		printf("  %s\n", log);
	}
	glAttachShader(m_screenquad_prog, vShader);
	glAttachShader(m_screenquad_prog, fShader);
	glLinkProgram(m_screenquad_prog);
	glGetProgramiv(m_screenquad_prog, GL_LINK_STATUS, &status);
	if (!status) {
		printf("*** Error! Failed to link in init_screenquad\n");
	}
	checkGL ( "glLinkProgram (init_screenquad)" );
	
	// Get texture parameter
	m_screenquad_utex1 = glGetUniformLocation (m_screenquad_prog, "uTex1" );
	m_screenquad_utex2 = glGetUniformLocation (m_screenquad_prog, "uTex2");
	m_screenquad_utexflags = glGetUniformLocation(m_screenquad_prog, "uTexFlags");
	m_screenquad_ucoords = glGetUniformLocation ( m_screenquad_prog, "uCoords" );
	m_screenquad_uscreen = glGetUniformLocation ( m_screenquad_prog, "uScreen" );


	// Create a screen-space quad VBO
	std::vector<nvVertex> verts;
	std::vector<nvFace> faces;
	verts.push_back(nvVertex(-1, -1, 0, -1, 1, 0));
	verts.push_back(nvVertex(1, -1, 0, 1, 1, 0));
	verts.push_back(nvVertex(1, 1, 0, 1, -1, 0));
	verts.push_back(nvVertex(-1, 1, 0, -1, -1, 0));
	faces.push_back(nvFace(0, 1, 2));
	faces.push_back(nvFace(2, 3, 0));

	glGenBuffers(1, (GLuint*)&m_screenquad_vbo[0]);
	glGenBuffers(1, (GLuint*)&m_screenquad_vbo[1]);
	checkGL("glGenBuffers (init_screenquad)");
	glGenVertexArrays(1, (GLuint*)&m_screenquad_vbo[2]);
	glBindVertexArray(m_screenquad_vbo[2]);
	checkGL("glGenVertexArrays (init_screenquad)");
	glBindBuffer(GL_ARRAY_BUFFER, m_screenquad_vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(nvVertex), &verts[0].x, GL_STATIC_DRAW_ARB);
	checkGL("glBufferData[V] (init_screenquad)");
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);				// pos
	glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);	// norm
	glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);	// texcoord
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenquad_vbo[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * 3 * sizeof(int), &faces[0].a, GL_STATIC_DRAW_ARB);
	checkGL("glBufferData[F] (init_screenquad)");
	glBindVertexArray(0);
}

void OptixApp::createScreenQuadGL ( int* glid, int w, int h ) {
	if ( *glid == -1 ) glDeleteTextures ( 1, (GLuint*) glid );
	glGenTextures ( 1, (GLuint*) glid );
	glBindTexture ( GL_TEXTURE_2D, *glid );
	checkGL ( "glBindTexture (createScreenQuadGL)" );
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 );	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);	
	glTexImage2D  ( GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	checkGL ( "glTexImage2D (createScreenQuadGL)" );
	glBindTexture ( GL_TEXTURE_2D, 0 );
}

void OptixApp::renderScreenQuadGL ( int glid1, int glid2, float x1, float y1, 
									float x2, float y2, char inv1, char inv2 ) {
	// Prepare pipeline
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(GL_FALSE);
	// Select shader	
	glBindVertexArray(m_screenquad_vbo[2]);
	glUseProgram(m_screenquad_prog);
	checkGL("glUseProgram");
	// Select VBO	
	glBindBuffer(GL_ARRAY_BUFFER, m_screenquad_vbo[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenquad_vbo[1]);
	checkGL("glBindBuffer");
	// Select texture
	
	//glEnable ( GL_TEXTURE_2D );
	
	glProgramUniform4f ( m_screenquad_prog, m_screenquad_ucoords, x1, y1, x2, y2 );
	glProgramUniform2f ( m_screenquad_prog, m_screenquad_uscreen, x2, y2 );
	glActiveTexture ( GL_TEXTURE0 );
	glBindTexture ( GL_TEXTURE_2D, glid1 );
    checkGL("glBindTexture");
	
	glProgramUniform1i(m_screenquad_prog, m_screenquad_utex1, 0);
	int flags = 0;
	if (inv1 > 0) flags |= 1;												// y-invert tex1

	if (glid2 >= 0) {
		flags |= 2;															// enable tex2 compositing
		if (inv2 > 0) flags |= 4;											// y-invert tex2
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, glid2);
		glProgramUniform1i(m_screenquad_prog, m_screenquad_utex2, 1);
	}

	glProgramUniform1i(m_screenquad_prog, m_screenquad_utexflags, flags );	

	// Draw
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1);
	
	checkGL("glDraw");
	glUseProgram(0);

	glDepthMask(GL_TRUE);
}

void OptixApp::renderScreenQuadGL( int glid, char inv1, int w, int h ) {
	renderScreenQuadGL ( glid, -1, (float)0, (float)0, (float)w, (float)h, inv1, 0); 
}