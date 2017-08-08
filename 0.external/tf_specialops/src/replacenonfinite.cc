#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_helper.h"

using namespace tensorflow;

REGISTER_OP("ReplaceNonfinite")
  .Attr("T: {float, double}")
  .Attr("value: float = 0.0")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
Replaces all nonfinite elements with 'value'.
The gradient for replaced elements is 0.


value:
  The value used for replacing nonfinite elements.

input: 
  Input tensor of any shape.

output:=Tensor with all nonfinite values replaced with 'value'.
)doc");



template <class T>
void replacenonfinite_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, 
    const T value,
    const int64_t size );


template <bool GPU, class T>
class ReplaceNonfiniteOp : public OpKernel 
{
public:
  explicit ReplaceNonfiniteOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    float value_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("value", &value_tmp));
    value = value_tmp;
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());

    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    const int64_t size = input_shape.num_elements();

    if(GPU)
    {
      OP_REQUIRES(context, (size/256+1) < std::numeric_limits<int>::max(),
          errors::Internal("Input size is too large for ReplaceNonfinite on gpu device"));
      auto device = context->eigen_gpu_device();
      replacenonfinite_gpu(
          device.stream(),
          output.data(),
          input.data(),
          value,
          size);
    }
    else
    {
      const T* in_ptr = input.data();
      T* out_ptr = output.data();
      for( int64_t i = 0; i < size; ++i )
      {
        const T tmp = in_ptr[i];
        out_ptr[i] = (std::isfinite(tmp) ? tmp : value);
      }
    }
    
  }

private:
  T value;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ReplaceNonfinite")                                                  \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ReplaceNonfiniteOp<false,type>);                                           
REG_KB(float)
REG_KB(double)
#undef REG_KB

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ReplaceNonfinite")                                                  \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ReplaceNonfiniteOp<true,type>);                                           
REG_KB(float)
REG_KB(double)
#undef REG_KB



REGISTER_OP("ReplaceNonfiniteGrad")
  .Attr("T: {float, double}")
  .Input("gradients: T")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
  .Doc(R"doc(
This computes the gradient for the op 'ReplaceNonfinite'. 
The gradient for nonfinite elements is 0.
)doc");


template <class T>
void replacenonfinite_grad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, const T* grad,
    const int64_t size );

template <bool GPU, class T>
class ReplaceNonfiniteGradOp : public OpKernel 
{
public:
  explicit ReplaceNonfiniteGradOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& gradients_tensor = context->input(0);
    auto gradients = gradients_tensor.flat<T>();
    const TensorShape gradients_shape(gradients_tensor.shape());
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());

    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    const int64_t size = input_shape.num_elements();

    if(GPU)
    {
      OP_REQUIRES(context, (size/256+1) < std::numeric_limits<int>::max(),
          errors::Internal("Input size is too large for ReplaceNonfiniteGrad on gpu device"));
      auto device = context->eigen_gpu_device();
      replacenonfinite_grad_gpu(
          device.stream(),
          output.data(),
          input.data(),
          gradients.data(),
          size);
    }
    else
    {
      const T* in_ptr = input.data();
      const T* grad_ptr = gradients.data();
      T* out_ptr = output.data();
      for( int64_t i = 0; i < size; ++i )
      {
        const T tmp = in_ptr[i];
        out_ptr[i] = (std::isfinite(tmp) ? grad_ptr[i] : T(0));
      }
    }
    
  }

private:
  T value;
};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ReplaceNonfiniteGrad")                                              \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ReplaceNonfiniteGradOp<false,type>);                                       
REG_KB(float)
REG_KB(double)
#undef REG_KB

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ReplaceNonfiniteGrad")                                              \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ReplaceNonfiniteGradOp<true,type>);                                       
REG_KB(float)
REG_KB(double)
#undef REG_KB

