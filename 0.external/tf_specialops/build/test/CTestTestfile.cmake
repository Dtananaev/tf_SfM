# CMake generated Testfile for 
# Source directory: /misc/student/tananaed/tfspecialops/test
# Build directory: /misc/student/tananaed/tfspecialops/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_AngleAxisToRotationMatrix "/misc/lmbraid11/tananaed/tf/bin/python" "test_AngleAxisToRotationMatrix.py")
set_tests_properties(test_AngleAxisToRotationMatrix PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_ReplaceNonfinite "/misc/lmbraid11/tananaed/tf/bin/python" "test_ReplaceNonfinite.py")
set_tests_properties(test_ReplaceNonfinite PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_Warp2d "/misc/lmbraid11/tananaed/tf/bin/python" "test_Warp2d.py")
set_tests_properties(test_Warp2d PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_ScaleInvariantGradient "/misc/lmbraid11/tananaed/tf/bin/python" "test_ScaleInvariantGradient.py")
set_tests_properties(test_ScaleInvariantGradient PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_MySqr "/misc/lmbraid11/tananaed/tf/bin/python" "test_MySqr.py")
set_tests_properties(test_MySqr PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_FlowToDepth "/misc/lmbraid11/tananaed/tf/bin/python" "test_FlowToDepth.py")
set_tests_properties(test_FlowToDepth PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_Median3x3Downsample "/misc/lmbraid11/tananaed/tf/bin/python" "test_Median3x3Downsample.py")
set_tests_properties(test_Median3x3Downsample PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_LeakyRelu "/misc/lmbraid11/tananaed/tf/bin/python" "test_LeakyRelu.py")
set_tests_properties(test_LeakyRelu PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_DepthToFlow "/misc/lmbraid11/tananaed/tf/bin/python" "test_DepthToFlow.py")
set_tests_properties(test_DepthToFlow PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_RotationMatrixToAngleAxis "/misc/lmbraid11/tananaed/tf/bin/python" "test_RotationMatrixToAngleAxis.py")
set_tests_properties(test_RotationMatrixToAngleAxis PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
add_test(test_DepthToNormals "/misc/lmbraid11/tananaed/tf/bin/python" "test_DepthToNormals.py")
set_tests_properties(test_DepthToNormals PROPERTIES  ENVIRONMENT "TFSPECIALOPS_LIB=/misc/student/tananaed/tfspecialops/build/lib/tfspecialops.so" WORKING_DIRECTORY "/misc/student/tananaed/tfspecialops/test")
