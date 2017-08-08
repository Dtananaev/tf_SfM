#ifndef HELPER_H
#define HELPER_H
#include "Eigen/Core"


template <class Derived>
inline bool matrix_is_finite( const Eigen::MatrixBase<Derived>& mat )
{
  for( int j = 0; j < mat.cols(); ++j )
  for( int i = 0; i < mat.rows(); ++i )
  {
    if( !std::isfinite(mat(i,j)) )
      return false;
  }
  return true;

}

#endif /* HELPER_H */
