#ifndef CUDA_HELPER_H_
#define CUDA_HELPER_H_
#include <iostream>
#include <stdexcept>

#define CHECK_CUDA_ERROR _CHECK_CUDA_ERROR(__FILE__, __LINE__);

inline void _CHECK_CUDA_ERROR( const char* filename, const int line )
{
  cudaError_t error = cudaGetLastError();
  if( error != cudaSuccess )
  {
    char str[1024]; str[0] = 0;
    sprintf(str, "%s:%d: cuda error: %s\n", 
           filename, line, cudaGetErrorString(error));
    throw std::runtime_error(str);
  }
}

inline void print_cuda_pointer_attributes( const void* ptr )
{
  cudaPointerAttributes attr = cudaPointerAttributes();
  cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
  char str[1024]; str[0] = 0;
  if( status != cudaSuccess )
  {
    sprintf(str, "cuda error in 'print_cuda_pointer_attributes()': %s", 
        cudaGetErrorString(status));
    //throw std::runtime_error(str);
  }
  std::cerr << "\n" << ptr << "  " << str << "\n";
  std::cerr << "memoryType: " 
    << (attr.memoryType==cudaMemoryTypeHost ? "cudaMemoryTypeHost\n" : "cudaMemoryTypeDevice\n") 
    << "device: " << attr.device << "\n"
    << "devicePointer: " << attr.devicePointer << "\n"
    << "hostPointer: " << attr.hostPointer << "\n"
    << "isManaged: " << attr.isManaged << "\n";
}



inline int divup( int x, int y )
{
  div_t tmp = std::div(x,y);
  return tmp.quot + (tmp.rem != 0 ? 1 : 0);
}

#endif /* CUDA_HELPER_H_ */
