#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("MySqr")
  .Attr("T: {float, double}")
  .Attr("bla: list(int) = [2,3]")
  .Attr("blub: string = 'hello world'")
  .Input("unsquared: T")
  .Output("squared: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
Squares all inputs
)doc");


template <class T>
class MySqrOp : public OpKernel 
{
public:
  explicit MySqrOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    OP_REQUIRES_OK(construction, construction->GetAttr("bla", &bla));
    std::cerr << "MySqrOp\n";
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<T>();

    const int N = input.size();
    for( int i = 0; i < N; ++i )
      output(i) = input(i) * input(i);

    //for( int i : bla )
      //std::cerr << i << "\n";
  }

private:
  std::vector<int> bla;

};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("MySqr")                                                             \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    MySqrOp<type>);                                                            
REG_KB(float)
REG_KB(double)
#undef REG_KB


template <class T>
void mysqr_gpu_kernel(const cudaStream_t& stream, T* out, const T* in, const int N);

template <class T>
class MySqrOpGPU : public OpKernel 
{
public:
  explicit MySqrOpGPU(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    OP_REQUIRES_OK(construction, construction->GetAttr("bla", &bla));
    OP_REQUIRES_OK(construction, construction->GetAttr("blub", &blub));
    std::cerr << "MySqrOpGPU\n";
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<T>();

    const int N = input.size();

    auto device = context->eigen_device<Eigen::GpuDevice>();
    //auto device = context->eigen_gpu_device();
    mysqr_gpu_kernel<T>(device.stream(), output.data(), input.data(), N);

    //for( int i : bla )
      //std::cerr << i << "\n";
    //std::cerr << blub << " gpu\n";
  }

private:
  std::vector<int> bla;
  std::string blub;

};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("MySqr")                                                             \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    MySqrOpGPU<type>);                                                         
REG_KB(float)
REG_KB(double)
#undef REG_KB






REGISTER_OP("MySqrGrad")
  .Attr("T: {float, double}")
  .Attr("bla: list(int) = [2,3]")
  .Attr("blub: string = 'hello world'")
  .Input("gradients: T")
  .Input("unsquared: T")
  .Output("backprops: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
This computes the gradient for MySqr
)doc");


template <class T>
void mysqrgrad_gpu_kernel(const cudaStream_t& stream, T* out, const T* in, const T* grad, const int N);

template <bool GPU, class T>
class MySqrGradOp : public OpKernel 
{
public:
  explicit MySqrGradOp(OpKernelConstruction* kernel_construction)
    :OpKernel(kernel_construction)
  { }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& gradient_tensor = context->input(0);
    auto gradient = gradient_tensor.flat<T>();
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<T>();

    const int N = input.size();
    if( GPU )
    {
      auto device = context->eigen_device<Eigen::GpuDevice>();
      //auto device = context->eigen_gpu_device();
      mysqrgrad_gpu_kernel(device.stream(), output.data(), input.data(), gradient.data(), N);
    }
    else
    {
      for( int i = 0; i < N; ++i )
      {
        output(i) = gradient(i) * 2*input(i);
        //std::cerr << output(i) << " = " << gradient(i) << " * 2* " << input(i) << "\n";
      }
    }
  }

};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("MySqrGrad")                                                         \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    MySqrGradOp<false,type>);                                                  
REG_KB(float)
REG_KB(double)
#undef REG_KB


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("MySqrGrad")                                                         \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    MySqrGradOp<true,type>);                                                   
REG_KB(float)
REG_KB(double)
#undef REG_KB




