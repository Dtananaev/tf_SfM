find_package( PythonInterp REQUIRED )
find_package( CUDA REQUIRED )
find_package( OpenMP REQUIRED )

include( "${CMAKE_SOURCE_DIR}/cmake/cuda_gencode_options.cmake" )
#list( APPEND CUDA_NVCC_FLAGS "-Xcompiler ${OpenMP_CXX_FLAGS}" )
list( APPEND CUDA_NVCC_FLAGS "-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0" )
list( APPEND CUDA_NVCC_FLAGS "${CUDA_GENCODE_STRING}" )
list( APPEND CUDA_NVCC_FLAGS_DEBUG -g -G -O0 --ptxas-options=-v )

# retrieve tensorflow include dir
execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" 
        OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR )
message( STATUS "${TENSORFLOW_INCLUDE_DIR}" )



file( GLOB tfspecialops_SOURCES  *.cc *.c *.cu )

include_directories( 
        ${TENSORFLOW_INCLUDE_DIR}
)

cuda_add_library( tfspecialops SHARED ${tfspecialops_SOURCES} )
# do not add 'lib' prefix
set_target_properties( tfspecialops PROPERTIES PREFIX "" )
set_target_properties( tfspecialops PROPERTIES DEBUG_POSTFIX "_debug" )
set_target_properties( tfspecialops PROPERTIES COMPILE_FLAGS "-std=c++11 ${OpenMP_CXX_FLAGS}" )
set_target_properties( tfspecialops PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}" )
# use old ABI with gcc 5.x
set_target_properties( tfspecialops PROPERTIES COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=0" )


