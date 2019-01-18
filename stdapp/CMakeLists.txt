cmake_minimum_required(VERSION 2.8.1) 

set(target_name stdapp)
project(${target_name})

# FIND OPENGL
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

include_directories(
    external/glm-0.9.4.0/
	external/glfw-3.0.3/include/GLFW/
	external/glew-1.9.0/include/
	../
    ../sutil/
	.
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include( ${CMAKE_MODULE_PATH}/Helpers.cmake )  
include( ${CMAKE_MODULE_PATH}/Packages.cmake )
include( ${CMAKE_MODULE_PATH}/FindOptix.cmake )  

####################################################################################
# Add CUDA
_add_package_CUDA()


## OPTIX
_add_package_OPTIX()
include_directories (
    ${OPTIX_INCLUDE_DIR}
)
link_directories(
    ${OPTIX_LIB_DIR}
)


add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

# Compile external dependencies 
add_subdirectory (external)

set( srcs
        ../app.h
        ../app.cpp
        GLUtils.h
        GLUtils.cpp
        Camera.h 
        Camera.cpp
        # sutil
        ../sutil/PPMLoader.cpp
        ../sutil/HDRLoader.cpp
        ../sutil/sutil.cpp
        # optix example
        ../tutorial.h
        ../tutorial10.cu
        ../box.cu
        ../commonStructs.h
        ../parallelogram.cu
        ../random.h
        # stdapp
        main.cpp 
        )

####################################################################################
# Compile PTX Files
#
message ( STATUS "Build CUDA kernels: ${srcs}" )
_COMPILEPTX ( SOURCES ${srcs} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128 )

add_executable(${target_name} ${srcs} )
target_link_libraries(${target_name} 
    ${OPENGL_LIBRARY}
	GLFW_303
	GLEW_190
    optix
)