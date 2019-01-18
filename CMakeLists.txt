cmake_minimum_required(VERSION 2.8.1) 
find_package(Omegalib)

# Set the module name here
set(target_name caveoptix)
SET(MODULE_NAME ${target_name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include( ${CMAKE_MODULE_PATH}/Helpers.cmake )  
include( ${CMAKE_MODULE_PATH}/Packages.cmake )  
include( ${CMAKE_MODULE_PATH}/FindOptix.cmake ) 

####################################################################################
# Add CUDA
#
_add_package_CUDA()


## OPTIX
_add_package_OPTIX()
include_directories (
    ${OPTIX_INCLUDE_DIR}
)
link_directories(
    ${OPTIX_LIB_DIR}
)


####################################################################################
# Compile PTX Files
#
file(GLOB CUDA_FILES RELATIVE "${BASE_DIRECTORY}" *.cu *.cuh)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )
_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128 )


include_directories(
        .
        ./sutil/
        ${OMEGA_INCLUDE_DIRS}
        /home/toand/git/omegalib/build/src/omicron/omicron/include
)

message(${CMAKE_SOURCE_DIR})
add_definitions(-DOMEGALIB_MODULE)

# Set module name and source files here
add_library(${MODULE_NAME} MODULE 
                # sutil
                ./sutil/PPMLoader.cpp
                ./sutil/HDRLoader.cpp
                ./sutil/sutil.cpp
                # optix example
                ./tutorial.h
                ./tutorial10.cu
                ./box.cu
                ./commonStructs.h
                ./parallelogram.cu
                ./random.h
                # app
                app.h
                app.cpp
		caveoptix.cpp 
        )

# Set the module library dependencies here
target_link_libraries(${MODULE_NAME}
        stdc++
        ${OMEGA_LIB}
        python2.7
        optix
        )

#------------------------------------------------------------------------------
# DO NOT MODIFY ANYTHING BELOW AFTER THIS LINE
set_target_properties(${MODULE_NAME} PROPERTIES PREFIX "")
if(WIN32)
        set_target_properties(${MODULE_NAME} PROPERTIES FOLDER modules SUFFIX ".pyd")
endif()