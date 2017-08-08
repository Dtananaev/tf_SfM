import tensorflow as tf
from tensorflow.python.framework import ops
import os

if 'TFSPECIALOPS_LIB' in os.environ:
    _lib_path = os.environ['TFSPECIALOPS_LIB']
else: # try to find the lib in the build directory relative to this file
    _lib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', 'build','lib', 'tfspecialops.so'))
if not os.path.isfile(_lib_path):
    raise ValueError('Cannot find tfspecialops.so . Set the environment variable TFSPECIALOPS_LIB to the path to tfspecialops.so file')
tfspecialopslib = tf.load_op_library(_lib_path)
tfspecialops = tfspecialopslib # TODO rm
#print('Using {0}'.format(_lib_path), flush=True)

# TODO create alias for each op in this module

@ops.RegisterGradient("MySqr")
def _my_sqr_grad(op, grad):
    return tfspecialopslib.my_sqr_grad(grad, op.inputs[0])


@ops.RegisterGradient("ScaleInvariantGradient")
def _scale_invariant_gradient_grad(op, grad):
    return tfspecialopslib.scale_invariant_gradient_grad(
            gradients=grad, 
            input=op.inputs[0], 
            deltas=op.get_attr('deltas'),
            weights=op.get_attr('weights'),
            epsilon=op.get_attr('epsilon') )


@ops.RegisterGradient("ReplaceNonfinite")
def _replace_nonfinite_grad(op, grad):
    return tfspecialopslib.replace_nonfinite_grad(
            gradients=grad, 
            input=op.inputs[0] )


@ops.RegisterGradient("LeakyRelu")
def _leaky_relu_grad(op, grad):
    return tfspecialopslib.leaky_relu_grad(
            gradients=grad, 
            input=op.inputs[0],
            leak=op.get_attr('leak'))


@ops.RegisterGradient("AngleAxisToRotationMatrix")
def _angle_axis_to_rotation_matrix_grad(op, grad):
    return tfspecialopslib.angle_axis_to_rotation_matrix_grad(
            gradients=grad, 
            in_=op.inputs[0], )


@ops.RegisterGradient("RotationMatrixToAngleAxis")
def _rotation_matrix_to_angle_axis_grad(op, grad):
    return tfspecialopslib.rotation_matrix_to_angle_axis_grad(
            gradients=grad, 
            in_=op.inputs[0], )



